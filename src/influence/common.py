"""Module with common methods for influence jobs"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pyspark.sql.window import Window
from pyspark.sql import functions as F
import airportsdata

DELAY_THRESHOLD = 15
NEW, OLD = 1, 0


def create_graph(data):
    """Create a directed graph from the flight data with delay probabilities."""
    route_delay_probability = (
        data
        .withColumn("is_delayed", F.when(F.col("DEPARTURE_DELAY") >
                                         DELAY_THRESHOLD, 1).otherwise(0))
        .groupBy("ORIGIN_AIRPORT", "DESTINATION_AIRPORT")
        .agg(
            F.count("*").alias("n_flights"),
            F.sum("is_delayed").alias("n_delayed")
        )
        .withColumn("delay_ratio", F.col("n_delayed") / F.col("n_flights"))
    )

    edges = route_delay_probability.select(
        "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "delay_ratio"
    ).collect()

    graph = {}
    for row in edges:
        origin = row["ORIGIN_AIRPORT"]
        dest = row["DESTINATION_AIRPORT"]
        prob = row["delay_ratio"]
        graph.setdefault(origin, []).append((dest, prob))

    graph = {
        origin: sorted(neighbors, key=lambda x: x[1], reverse=True)
        for origin, neighbors in graph.items()
    }

    return graph


def scheduled_departure_minutes(hhmm, airport, airport_tz, year, month, day):
    """Absolute instant of a local HHMM departure, in minutes.
    The airport's time zone is applied, so the value is comparable across
    airports and across days."""
    if hhmm is None:
        return None
    hour, minute = divmod(int(hhmm), 100)
    tz = ZoneInfo(airport_tz.get(airport, "America/Chicago"))
    local_dt = (datetime(year, month, day, tzinfo=tz)
                + timedelta(hours=hour, minutes=minute))  # also handles 2400
    return int(local_dt.timestamp() // 60)


def create_graph_v2(data, max_gap_minutes=360, min_pairs=20):
    """Directed graph origin -> destination, weighted by how often the delay
    grows on the next leg of the same aircraft."""
    spark = data.sparkSession

    airports = airportsdata.load("IATA")
    airport_tz = {code: info["tz"] for code, info in airports.items() if info.get("tz")}
    airport_tz["ISN"] = "America/Chicago"
    airport_tz_bc = spark.sparkContext.broadcast(airport_tz)

    sched_ts = F.udf(
        lambda hhmm, airport, y, m, d: scheduled_departure_minutes(
            hhmm, airport, airport_tz_bc.value, y, m, d),
        "long")

    flights = (
        data
        .filter((F.col("CANCELLED") == 0)
                & F.col("TAIL_NUMBER").isNotNull()
                & (F.col("TAIL_NUMBER") != "")
                & F.col("DEPARTURE_DELAY").isNotNull())
        .withColumn("sched_ts", sched_ts(
            F.col("SCHEDULED_DEPARTURE"), F.col("ORIGIN_AIRPORT"),
            F.col("YEAR"), F.col("MONTH"), F.col("DAY")))
        .filter(F.col("sched_ts").isNotNull())
        .select("TAIL_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
                "DEPARTURE_DELAY", "sched_ts")
    )

    # legs of each aircraft, in real chronological order
    w = Window.partitionBy("TAIL_NUMBER").orderBy("sched_ts")

    pairs = (
        flights
        .withColumn("next_origin", F.lead("ORIGIN_AIRPORT").over(w))
        .withColumn("next_delay", F.lead("DEPARTURE_DELAY").over(w))
        .withColumn("next_ts", F.lead("sched_ts").over(w))
        # the legs must chain: arrive at YYY, then leave from YYY
        .filter(F.col("next_origin") == F.col("DESTINATION_AIRPORT"))
    )

    if max_gap_minutes is not None:
        pairs = pairs.filter(F.col("next_ts") - F.col("sched_ts") <= max_gap_minutes)

    stats = (
        pairs
        .withColumn("increased",
                    F.when(F.col("next_delay")
                           > F.greatest(F.col("DEPARTURE_DELAY"), F.lit(0)), 1)
                     .otherwise(0))
        .groupBy("ORIGIN_AIRPORT", "DESTINATION_AIRPORT")
        .agg(F.count("*").alias("n_pairs"),
             F.sum("increased").alias("n_increased"))
    )

    # keep every route of the dataset
    routes = data.select("ORIGIN_AIRPORT", "DESTINATION_AIRPORT").distinct()

    edges = (
        routes
        .join(stats, ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], "left")
        .withColumn("prob",
                    F.when(F.col("n_pairs") >= min_pairs,
                           F.col("n_increased") / F.col("n_pairs"))
                     .otherwise(F.lit(0.0)))
        .select("ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "prob")
        .collect()
    )

    graph = {}
    for row in edges:
        graph.setdefault(row["ORIGIN_AIRPORT"], []).append(
            (row["DESTINATION_AIRPORT"], float(row["prob"])))

    return {
        origin: sorted(neighbors, key=lambda x: x[1], reverse=True)
        for origin, neighbors in graph.items()
    }


def merge_labels(label1, label2):
    """Merge labels for the same node: OLD prevails."""
    if label1 == OLD or label2 == OLD:
        return OLD
    return NEW
