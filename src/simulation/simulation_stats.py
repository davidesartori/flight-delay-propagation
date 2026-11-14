"""Module for computing simulation statistics."""

from pyspark.sql import functions as F


def get_magnitude(flight_info_bc, state_rdd, t: int):
    """Calculate the magnitude of delays at each airport."""
    state_rdd = state_rdd.filter(
        lambda flight: flight[1]["is_departed"] is True and flight[1]["departure_time"] < t and
        flight[1]["departure_time"] >= t-60)
    delay_time_per_airport = state_rdd.map(lambda flight:
                                           (flight_info_bc.value[flight[0]]["origin"],
                                            max(flight[1]["departure_time"] -
                                                flight[1]["scheduled_departure"],
                                                0))).reduceByKey(lambda a, b:
                                                                 a + b).filter(lambda item:
                                                                                 item[1] != 0)

    delay_count_per_airport = state_rdd.map(lambda flight: (flight_info_bc.value[flight[0]]["origin"],
                                                            1 if flight[1]["departure_time"] -
                                                            flight[1]["scheduled_departure"] >= 60 else
                                                            0)).reduceByKey(lambda a, b:
                                                                            a + b).filter(lambda item: item[1] != 0)

    return delay_count_per_airport, delay_time_per_airport


def get_speed(magnitude_rdd, n_total_airports: int):
    """Computes the speed of delay propagation for each airport."""
    total = magnitude_rdd.map(lambda x: x[1]).sum()

    return total / n_total_airports if n_total_airports > 0 else 0


def top_airports_by_delay(df, airport_col, delay_col, top_n=10, min_flights=5):
    """Ranks airports by total delay (only delays >= 1 min count toward the total)."""
    ranking = (
        df.groupBy(airport_col)
          .agg(
              F.count("*").alias("n_flights"),
              F.count(F.when(F.col(delay_col) >= 15, 1)).alias("n_delayed"),
              F.sum(F.when(F.col(delay_col) >= 15, F.col(delay_col)).otherwise(0)).alias("total_delay"),
              F.count(F.when(F.col(delay_col) >= 60, 1)).alias("n_delays_gt_60"),
              F.count(F.when(F.col(delay_col) >= 120, 1)).alias("n_delays_gt_120"),
              F.count(F.when(F.col(delay_col) >= 200, 1)).alias("n_delays_gt_200"),
              F.count(F.when(F.col(delay_col) >= 250, 1)).alias("n_delays_gt_250"),
              F.count(F.when(F.col(delay_col) >= 300, 1)).alias("n_delays_gt_300"),
              F.count(F.when(F.col(delay_col) >= 400, 1)).alias("n_delays_gt_400"),
              F.avg(delay_col).alias("avg_delay"),
              F.max(delay_col).alias("max_delay"),
          )
          .filter(F.col("n_flights") >= min_flights)
          .orderBy(F.col("total_delay").desc())
          .limit(top_n)
          .collect()
    )
    return [
        {
            "airport": r[airport_col],
            "n_flights": r["n_flights"],
            "n_delayed": r["n_delayed"],
            "total_delay": r["total_delay"],
            "n_delays_gt_60": r["n_delays_gt_60"],
            "n_delays_gt_120": r["n_delays_gt_120"],
            "n_delays_gt_200": r["n_delays_gt_200"],
            "n_delays_gt_250": r["n_delays_gt_250"],
            "n_delays_gt_300": r["n_delays_gt_300"],
            "n_delays_gt_400": r["n_delays_gt_400"],
            "avg_delay": float(r["avg_delay"]),
            "max_delay": r["max_delay"],
        }
        for r in ranking
    ]


def compute_simulation_stats(stat_df, influence_scores, top_n=20):
    """Computes statistics from the results DataFrame."""
    not_departed_count = stat_df.filter(F.col("actual_departure").isNull()).count()

    df = (stat_df
        .filter(F.col("actual_arrival").isNotNull())
        .withColumn("delay", F.col("actual_arrival") - F.col("scheduled_arrival"))
        .withColumn("departure_delay",
                    F.col("actual_departure") - F.col("scheduled_departure"))
        .cache())

    n_departed = df.count()
    if n_departed == 0:
        return empty_stats(not_departed_count)

    thresholds = [60, 120, 200, 250, 300, 400]
    agg = df.agg(
        F.sum(F.when(F.col("delay") >= 1, F.col("delay")).otherwise(0)).alias("total_delay"),
        F.count(F.when(F.col("delay") >= 1, 1)).alias("n_delayed"),
        F.max("delay").alias("max_delay"),
        *[F.count(F.when(F.col("delay") >= th, 1)).alias(f"gt_{th}") for th in thresholds],
    ).first()

    total_delay = agg["total_delay"] or 0
    n_delayed = agg["n_delayed"]

    all_delays_sorted = sorted(r["delay"] for r in df.select("delay").collect())
    percentiles = {f"p{p}": percentile(all_delays_sorted, p) for p in [50, 75, 90, 95, 99]}

    ranked = sorted(influence_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_airports = [a for a, _ in ranked[:top_n]]

    top_df = df.filter(F.col("destination").isin(top_airports))
    top_delays_sorted = sorted(r["delay"] for r in top_df.select("delay").collect())
    n_top = len(top_delays_sorted)

    if n_top > 0:
        top_avg_delay = sum(top_delays_sorted) / n_top
        top_median_delay = percentile(top_delays_sorted, 50)
        top_total_delay = sum(d for d in top_delays_sorted if d >= 1)
    else:
        top_avg_delay = top_median_delay = top_total_delay = 0

    weights = df.sparkSession.createDataFrame(
        [(k, float(v)) for k, v in influence_scores.items()],
        "destination string, influence double",
    )
    
    total_weighted_delay = (
        df.filter(F.col("delay") >= 1)
          .join(weights, "destination", "left")
          .agg(F.sum(F.col("delay") * F.coalesce(F.col("influence"), F.lit(0))))
          .first()[0] or 0
    )

    top_delays = top_airports_by_delay(df, "origin", "departure_delay", top_n=top_n)

    return {
        "total_delay": total_delay,
        "n_delayed_flights": n_delayed,
        **{f"n_delays_gt_{th}": agg[f"gt_{th}"] for th in thresholds},
        "n_departed_flights": n_departed,
        "n_not_departed": not_departed_count,
        "avg_delay": total_delay / n_departed,
        "avg_delay_delayed": total_delay / n_delayed if n_delayed > 0 else 0,
        "median_delay": percentiles["p50"],
        "percentiles": percentiles,
        "max_delay": agg["max_delay"],
        "n_top_airport_flights": n_top,
        "top_avg_delay": top_avg_delay,
        "top_median_delay": top_median_delay,
        "top_total_delay": top_total_delay,
        "total_weighted_delay": total_weighted_delay,
        "top_delays": top_delays,
    }


def percentile(sorted_values, p):
    """Returns the percentile calculated over the given values"""
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, int(round((p / 100) * (len(sorted_values) - 1))))
    return sorted_values[idx]


def empty_stats(not_departed_count: int):
    """Returns stats when no flight is departed"""
    empty_percentiles = {f"p{p}": 0 for p in [50, 75, 90, 95, 99]}
    return {
        "total_delay": 0, "n_delayed_flights": 0,
        **{f"n_delays_gt_{th}": 0 for th in [60, 120, 200, 250, 300, 400]},
        "n_departed_flights": 0,
        "n_not_departed": not_departed_count, "avg_delay": 0,
        "avg_delay_delayed": 0, "median_delay": 0,
        "percentiles": empty_percentiles, "max_delay": 0,
        "n_top_airport_flights": 0, "top_avg_delay": 0,
        "top_median_delay": 0, "top_total_delay": 0, "total_weighted_delay": 0,
        "top_delays": [],
    }
