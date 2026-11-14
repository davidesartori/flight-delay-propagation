"""Tests for the simulation statistics module."""
import pytest
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from src.simulation.simulation_stats import (
    get_magnitude,
    get_speed,
    compute_simulation_stats,
    top_airports_by_delay,
    percentile,
    empty_stats,
)

STAT_SCHEMA = StructType([
    StructField("flight_id", StringType(), False),
    StructField("tail_number", StringType(), True),
    StructField("origin", StringType(), True),
    StructField("destination", StringType(), True),
    StructField("scheduled_departure", IntegerType(), True),
    StructField("actual_departure", IntegerType(), True),
    StructField("scheduled_arrival", IntegerType(), True),
    StructField("actual_arrival", IntegerType(), True),
])


def flight(flight_id, origin="ATL", destination="ORD", scheduled_departure=0, actual_departure=0, scheduled_arrival=60, actual_arrival=60):
    """A departed and landed flight (one row of the results DataFrame)."""
    return (flight_id, "N1", origin, destination, scheduled_departure, actual_departure, scheduled_arrival, actual_arrival)


def not_departed(flight_id, origin="ATL", destination="ORD", scheduled_departure=0, scheduled_arrival=60):
    """A flight that never departed: actual times are null."""
    return (flight_id, "N1", origin, destination, scheduled_departure, None, scheduled_arrival, None)


def make_stat_df(spark, rows):
    """Build a results DataFrame with an explicit schema (nulls are fine)."""
    return spark.createDataFrame(rows, schema=STAT_SCHEMA)


def test_percentile_empty_list():
    """Return zero when computing a percentile of an empty list."""
    assert percentile([], 50) == 0


def test_percentile_single_value():
    """Return the only value for any percentile of a single-value list."""
    assert percentile([42], 50) == 42
    assert percentile([42], 99) == 42


def test_percentile_basic():
    """Compute the median correctly for a basic ordered dataset."""
    assert percentile([10, 20, 30, 40, 50], 50) == 30


def test_empty_stats_structure():
    """Return the expected default statistics structure for empty results."""
    stats = empty_stats(not_departed_count=7)

    assert stats["n_not_departed"] == 7
    assert stats["n_departed_flights"] == 0
    assert stats["total_delay"] == 0
    assert stats["percentiles"] == {"p50": 0, "p75": 0, "p90": 0, "p95": 0, "p99": 0}
    assert not stats["top_delays"]


def test_empty_stats_has_same_keys_as_compute(spark):
    """save_stats reads the same keys in both cases, so they must match."""
    df = make_stat_df(spark, [flight("F1")])
    full = compute_simulation_stats(df, influence_scores={"ORD": 1})
    assert set(empty_stats(0)) == set(full)


def test_get_magnitude_counts_and_sums_correctly(spark):
    """Count delayed flights and sum their delays correctly per airport."""
    sc = spark.sparkContext

    flight_info = {
        "F1": {"origin": "ATL"},
        "F2": {"origin": "ATL"},
        "F3": {"origin": "ORD"},
    }
    flight_info_bc = sc.broadcast(flight_info)

    state_rdd = sc.parallelize([
        ("F1", {"is_departed": True, "departure_time": 100, "scheduled_departure": 30}),
        ("F2", {"is_departed": True, "departure_time": 110, "scheduled_departure": 100}),
        ("F3", {"is_departed": True, "departure_time": 90, "scheduled_departure": 90}),
    ])

    count_rdd, time_rdd = get_magnitude(flight_info_bc, state_rdd, t=120)

    assert dict(count_rdd.collect()) == {"ATL": 1}
    assert dict(time_rdd.collect()) == {"ATL": 80}


def test_get_magnitude_excludes_flights_outside_window(spark):
    """Exclude flights whose departure time falls outside the window."""
    sc = spark.sparkContext
    flight_info_bc = sc.broadcast({"F1": {"origin": "ATL"}})

    state_rdd = sc.parallelize([
        ("F1", {"is_departed": True, "departure_time": 10, "scheduled_departure": 0}),
    ])

    count_rdd, time_rdd = get_magnitude(flight_info_bc, state_rdd, t=120)

    assert count_rdd.collect() == []
    assert time_rdd.collect() == []


def test_get_magnitude_excludes_not_departed_flights(spark):
    """Exclude flights that have not departed from the magnitude calculation."""
    sc = spark.sparkContext
    flight_info_bc = sc.broadcast({"F1": {"origin": "ATL"}})

    state_rdd = sc.parallelize([
        ("F1", {"is_departed": False, "departure_time": 100, "scheduled_departure": 30}),
    ])

    count_rdd, time_rdd = get_magnitude(flight_info_bc, state_rdd, t=120)

    assert count_rdd.collect() == []
    assert time_rdd.collect() == []


def test_get_speed_basic(spark):
    """Compute the average magnitude across the total number of airports."""
    magnitude_rdd = spark.sparkContext.parallelize([("ATL", 8), ("ORD", 4), ("DEN", 0)])
    assert get_speed(magnitude_rdd, n_total_airports=4) == (8 + 4 + 0) / 4


def test_get_speed_zero_airports_returns_zero(spark):
    """Return zero when the total number of airports is zero."""
    magnitude_rdd = spark.sparkContext.parallelize([("ATL", 8)])
    assert get_speed(magnitude_rdd, n_total_airports=0) == 0


def test_get_speed_empty_rdd(spark):
    """Return zero when the magnitude RDD is empty."""
    magnitude_rdd = spark.sparkContext.parallelize([], numSlices=1)
    assert get_speed(magnitude_rdd, n_total_airports=5) == 0


def _delay_df(spark, rows):
    """DataFrame with (origin, departure_delay) rows."""
    return spark.createDataFrame(rows, ["origin", "departure_delay"])


def test_top_airports_by_delay_aggregates(spark):
    """Counts and sums use the 15 and 60 minute thresholds."""
    delays = [0, 15, 60, 100, 5]
    df = _delay_df(spark, [("ATL", d) for d in delays])

    (row,) = top_airports_by_delay(df, "origin", "departure_delay")

    assert row["airport"] == "ATL"
    assert row["n_flights"] == 5
    assert row["n_delayed"] == 3
    assert row["total_delay"] == 175
    assert row["n_delays_gt_60"] == 2
    assert row["avg_delay"] == pytest.approx(36.0)
    assert row["max_delay"] == 100


def test_top_airports_by_delay_orders_by_total_and_limits(spark):
    """Airports are sorted by total delay (descending) and cut to top_n."""
    rows = ([("ATL", 30)] * 5) + ([("ORD", 100)] * 5) + ([("DEN", 0)] * 5)
    df = _delay_df(spark, rows)

    ranking = top_airports_by_delay(df, "origin", "departure_delay", top_n=2)

    assert [r["airport"] for r in ranking] == ["ORD", "ATL"]


def test_top_airports_by_delay_applies_min_flights(spark):
    """Airports with fewer than min_flights flights are excluded."""
    rows = [("ATL", 10)] * 5 + [("DEN", 500)] * 2
    df = _delay_df(spark, rows)

    ranking = top_airports_by_delay(df, "origin", "departure_delay", min_flights=5)

    assert [r["airport"] for r in ranking] == ["ATL"]


def test_compute_simulation_stats_no_departed_flights(spark):
    """Verify statistics when no flights have departed."""
    df = make_stat_df(spark, [not_departed("F1"), not_departed("F2")])

    stats = compute_simulation_stats(df, influence_scores={})

    assert stats["n_departed_flights"] == 0
    assert stats["n_not_departed"] == 2
    assert stats["total_delay"] == 0


def test_compute_simulation_stats_counts_not_departed(spark):
    """Flights with a null actual_departure are counted as not departed."""
    df = make_stat_df(spark, [flight("F1"), not_departed("F2")])

    stats = compute_simulation_stats(df, influence_scores={})

    assert stats["n_departed_flights"] == 1
    assert stats["n_not_departed"] == 1


def test_compute_simulation_stats_basic_delays(spark):
    """Verify basic delay statistics (delay = actual_arrival - scheduled_arrival)."""
    df = make_stat_df(spark, [
        flight("F1", destination="ORD", scheduled_arrival=60, actual_arrival=130),
        flight("F2", destination="DEN", scheduled_arrival=190, actual_arrival=190),
    ])

    stats = compute_simulation_stats(df, influence_scores={})

    assert stats["n_departed_flights"] == 2
    assert stats["n_delayed_flights"] == 1
    assert stats["total_delay"] == 70
    assert stats["avg_delay"] == 70 / 2
    assert stats["avg_delay_delayed"] == 70 / 1
    assert stats["max_delay"] == 70
    assert stats["top_delays"] == []


def test_compute_simulation_stats_early_arrivals_do_not_reduce_total(spark):
    """Negative delays are not delays: only delays >= 1 count in the total."""
    df = make_stat_df(spark, [
        flight("F1", scheduled_arrival=100, actual_arrival=90),
        flight("F2", scheduled_arrival=100, actual_arrival=120),
    ])

    stats = compute_simulation_stats(df, influence_scores={})

    assert stats["total_delay"] == 20
    assert stats["n_delayed_flights"] == 1
    assert stats["max_delay"] == 20


def test_compute_simulation_stats_percentiles(spark):
    """Percentiles are computed over all departed flights, delayed or not."""
    df = make_stat_df(spark, [
        flight(f"F{i}", scheduled_arrival=0, actual_arrival=d) for i, d in enumerate([0, 10, 20, 30, 40])
    ])

    stats = compute_simulation_stats(df, influence_scores={})

    assert stats["median_delay"] == 20
    assert stats["percentiles"]["p50"] == 20
    assert stats["percentiles"]["p99"] == 40


def test_compute_simulation_stats_delay_thresholds(spark):
    """Verify the number of flights at or above each delay threshold."""
    delays = {"F60": 60, "F120": 120, "F200": 200,
              "F250": 250, "F300": 300, "F400": 400}
    df = make_stat_df(spark, [flight(fid, scheduled_arrival=0, actual_arrival=d) for fid, d in delays.items()])

    stats = compute_simulation_stats(df, influence_scores={})

    assert stats["n_delays_gt_60"] == 6
    assert stats["n_delays_gt_120"] == 5
    assert stats["n_delays_gt_200"] == 4
    assert stats["n_delays_gt_250"] == 3
    assert stats["n_delays_gt_300"] == 2
    assert stats["n_delays_gt_400"] == 1


def test_compute_simulation_stats_top_airports_weighting(spark):
    """Verify delay statistics and influence weighting for top airports."""
    df = make_stat_df(spark, [
        flight("F1", destination="ATL", scheduled_arrival=0, actual_arrival=50),
        flight("F2", destination="ORD", scheduled_arrival=0, actual_arrival=30),
    ])

    stats = compute_simulation_stats(
        df, influence_scores={"ATL": 90, "ORD": 10}, top_n=1
    )

    assert stats["n_top_airport_flights"] == 1
    assert stats["top_avg_delay"] == 50
    assert stats["top_total_delay"] == 50
    assert stats["total_weighted_delay"] == pytest.approx((50 * 90) + (30 * 10))


def test_compute_simulation_stats_destination_without_score_has_zero_weight(spark):
    """Destinations missing from influence_scores contribute nothing."""
    df = make_stat_df(spark, [
        flight("F1", destination="ATL", scheduled_arrival=0, actual_arrival=50),
        flight("F2", destination="XXX", scheduled_arrival=0, actual_arrival=30),
    ])

    stats = compute_simulation_stats(df, influence_scores={"ATL": 2})

    assert stats["total_weighted_delay"] == pytest.approx(100)


def test_compute_simulation_stats_no_delayed_flights(spark):
    """Verify statistics when all departed flights have zero delay."""
    df = make_stat_df(spark, [flight("F1", scheduled_arrival=60, actual_arrival=60)])

    stats = compute_simulation_stats(df, influence_scores={})

    assert stats["n_departed_flights"] == 1
    assert stats["n_delayed_flights"] == 0
    assert stats["total_delay"] == 0
    assert stats["avg_delay"] == 0
    assert stats["avg_delay_delayed"] == 0


def test_compute_simulation_stats_top_delay_origins_uses_departure_delay(spark):
    """The origin ranking uses actual_departure - scheduled_departure."""
    departure_delays = [0, 15, 60, 100, 5]
    df = make_stat_df(spark, [
        flight(f"F{i}", origin="ATL", scheduled_departure=0, actual_departure=d, scheduled_arrival=0, actual_arrival=0)
        for i, d in enumerate(departure_delays)
    ])

    stats = compute_simulation_stats(df, influence_scores={})

    (row,) = stats["top_delays"]
    assert row["airport"] == "ATL"
    assert row["n_flights"] == 5
    assert row["total_delay"] == 175
    assert row["max_delay"] == 100
