"""Tests for the simulation statistics module."""
from src.simulation.simulation_stats import (
    get_magnitude,
    get_speed,
    compute_simulation_stats,
    percentile,
    empty_stats,
)


def test_percentile_empty_list():
    """Return zero when computing a percentile of an empty list."""
    assert percentile([], 50) == 0


def test_percentile_single_value():
    """Return the only value for any percentile of a single-value list."""
    assert percentile([42], 50) == 42
    assert percentile([42], 99) == 42


def test_percentile_basic():
    """Compute the median correctly for a basic ordered dataset."""
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30


def test_empty_stats_structure():
    """Return the expected default statistics structure for empty results."""
    stats = empty_stats(not_departed_count=7)

    assert stats["n_not_departed"] == 7
    assert stats["n_departed_flights"] == 0
    assert stats["total_delay"] == 0
    assert stats["percentiles"] == {"p50": 0, "p75": 0, "p90": 0, "p95": 0, "p99": 0}


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

    delay_count_per_airport, delay_time_per_airport = get_magnitude(flight_info_bc, state_rdd, t=120)

    count_result = dict(delay_count_per_airport.collect())
    time_result = dict(delay_time_per_airport.collect())

    assert count_result == {"ATL": 1}
    assert time_result == {"ATL": 80}


def test_get_magnitude_excludes_flights_outside_window(spark):
    """Exclude flights whose departure time falls outside the simulation window."""
    sc = spark.sparkContext
    flight_info_bc = sc.broadcast({"F1": {"origin": "ATL"}})

    state_rdd = sc.parallelize([
        ("F1", {"is_departed": True, "departure_time": 10, "scheduled_departure": 0}),
    ])

    delay_count_per_airport, delay_time_per_airport = get_magnitude(flight_info_bc, state_rdd, t=120)

    assert delay_count_per_airport.collect() == []
    assert delay_time_per_airport.collect() == []


def test_get_magnitude_excludes_not_departed_flights(spark):
    """Exclude flights that have not departed from the magnitude calculation."""
    sc = spark.sparkContext
    flight_info_bc = sc.broadcast({"F1": {"origin": "ATL"}})

    state_rdd = sc.parallelize([
        ("F1", {"is_departed": False, "departure_time": 100, "scheduled_departure": 30}),
    ])

    delay_count_per_airport, delay_time_per_airport = get_magnitude(flight_info_bc, state_rdd, t=120)

    assert delay_count_per_airport.collect() == []
    assert delay_time_per_airport.collect() == []


def test_get_speed_basic(spark):
    """Compute the average magnitude across the total number of airports."""
    sc = spark.sparkContext
    magnitude_rdd = sc.parallelize([("ATL", 8), ("ORD", 4), ("DEN", 0)])

    speed = get_speed(magnitude_rdd, n_total_airports=4)

    assert speed == (8 + 4 + 0) / 4


def test_get_speed_zero_airports_returns_zero(spark):
    """Return zero when the total number of airports is zero."""
    sc = spark.sparkContext
    magnitude_rdd = sc.parallelize([("ATL", 8)])

    speed = get_speed(magnitude_rdd, n_total_airports=0)

    assert speed == 0


def test_get_speed_empty_rdd(spark):
    """Return zero when the magnitude RDD is empty."""
    sc = spark.sparkContext
    magnitude_rdd = sc.parallelize([], numSlices=1)

    speed = get_speed(magnitude_rdd, n_total_airports=5)

    assert speed == 0


def _make_flight_info_bc(sc, mapping):
    """Create a Spark broadcast variable containing flight information.

    Args:
        sc: SparkContext used to create the broadcast variable.
        mapping: Mapping from flight ID to a tuple containing the destination
            airport and scheduled flying time.

    Returns:
        A Spark broadcast variable containing flight information.
    """
    flight_info = {
        fid: {"destination": dest, "scheduled_flying_time": sft}
        for fid, (dest, sft) in mapping.items()
    }
    return sc.broadcast(flight_info)


def test_compute_simulation_stats_no_departed_flights(spark):
    """Verify statistics when no flights have departed."""
    sc = spark.sparkContext
    state_rdd = sc.parallelize([
        ("F1", {"is_departed": False}),
        ("F2", {"is_departed": False}),
    ])
    flight_info_bc = _make_flight_info_bc(sc, {"F1": ("ORD", 60), "F2": ("DEN", 90)})

    stats = compute_simulation_stats(state_rdd, flight_info_bc, influence_scores={})

    assert stats["n_departed_flights"] == 0
    assert stats["n_not_departed"] == 2
    assert stats["total_delay"] == 0


def test_compute_simulation_stats_basic_delays(spark):
    """Verify basic delay statistics for a set of departed flights."""
    sc = spark.sparkContext
    state_rdd = sc.parallelize([
        ("F1", {"is_departed": True, "scheduled_departure": 0, "scheduled_arrival": 130}),
        ("F2", {"is_departed": True, "scheduled_departure": 100, "scheduled_arrival": 190}),
    ])
    flight_info_bc = _make_flight_info_bc(sc, {
        "F1": ("ORD", 60),
        "F2": ("DEN", 90),
    })

    stats = compute_simulation_stats(state_rdd, flight_info_bc, influence_scores={})

    assert stats["n_departed_flights"] == 2
    assert stats["n_delayed_flights"] == 1
    assert stats["total_delay"] == 70
    assert stats["avg_delay"] == 70 / 2
    assert stats["avg_delay_delayed"] == 70 / 1
    assert stats["max_delay"] == 70


def test_compute_simulation_stats_delay_thresholds(spark):
    """Verify the number of flights exceeding each delay threshold."""
    sc = spark.sparkContext

    def flight(fid, delay):
        return (fid, {"is_departed": True, "scheduled_departure": 0, "scheduled_arrival": delay})

    state_rdd = sc.parallelize([
        flight("F60", 60),
        flight("F120", 120),
        flight("F200", 200),
        flight("F250", 250),
        flight("F300", 300),
        flight("F400", 400),
    ])
    flight_info_bc = _make_flight_info_bc(sc, {
        fid: ("ORD", 0) for fid in ["F60", "F120", "F200", "F250", "F300", "F400"]
    })

    stats = compute_simulation_stats(state_rdd, flight_info_bc, influence_scores={})

    assert stats["n_delays_gt_60"] == 6
    assert stats["n_delays_gt_120"] == 5
    assert stats["n_delays_gt_200"] == 4
    assert stats["n_delays_gt_250"] == 3
    assert stats["n_delays_gt_300"] == 2
    assert stats["n_delays_gt_400"] == 1


def test_compute_simulation_stats_top_airports_weighting(spark):
    """Verify delay statistics and influence weighting for top airports."""
    sc = spark.sparkContext

    state_rdd = sc.parallelize([
        ("F1", {"is_departed": True, "scheduled_departure": 0, "scheduled_arrival": 50}),
        ("F2", {"is_departed": True, "scheduled_departure": 0, "scheduled_arrival": 30}),
    ])
    flight_info_bc = _make_flight_info_bc(sc, {
        "F1": ("ATL", 0),
        "F2": ("ORD", 0),
    })

    influence_scores = {"ATL": 90, "ORD": 10}

    stats = compute_simulation_stats(
        state_rdd, flight_info_bc, influence_scores, top_n=1
    )

    assert stats["n_top_airport_flights"] == 1
    assert stats["top_avg_delay"] == 50
    assert stats["top_total_delay"] == 50

    assert stats["total_weighted_delay"] == (50 * 90) + (30 * 10)


def test_compute_simulation_stats_no_delayed_flights(spark):
    """Verify statistics when all departed flights have zero delay."""
    sc = spark.sparkContext

    state_rdd = sc.parallelize([
        ("F1", {"is_departed": True, "scheduled_departure": 0, "scheduled_arrival": 60}),
    ])
    flight_info_bc = _make_flight_info_bc(sc, {"F1": ("ORD", 60)})

    stats = compute_simulation_stats(state_rdd, flight_info_bc, influence_scores={})

    assert stats["n_departed_flights"] == 1
    assert stats["n_delayed_flights"] == 0
    assert stats["total_delay"] == 0
    assert stats["avg_delay"] == 0
    assert stats["avg_delay_delayed"] == 0
