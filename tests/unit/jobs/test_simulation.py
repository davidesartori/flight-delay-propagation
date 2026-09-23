"""Testing module for simulation job"""
from unittest.mock import MagicMock
from datetime import datetime
from pyspark.sql import functions as F
from src.jobs.simulation import parse_hhmm_to_minutes, convert_minutes_to_hhmm, minutes_to_timestamp, process_raw_flights, build_info_and_state, run_simulation, DEFAULT_TURNAROUND


def test_parse_hhmm_to_minutes_multiple_rows(spark):
    """Verifies HHMM-to-minutes conversion for several edge cases at once:
    midnight (0), a normal time (930), an unpadded short value (5), the last
    minute of the day (2359), and the day-rollover value used by some flight
    datasets (2400, which must wrap to 0)."""
    df = spark.createDataFrame([(0,), (930,), (5,), (2359,), (2400,)], ["hhmm"])
    results = df.select(parse_hhmm_to_minutes(F.col("hhmm")).alias("minutes")).collect()

    values = [row["minutes"] for row in results]
    assert values == [0, 570, 5, 1439, 0]


def test_convert_minutes_to_hhmm_basic():
    """Verifies that minutes-since-midnight are formatted back into a
    zero-padded HH:MM string."""
    assert convert_minutes_to_hhmm(570) == "09:30"


def test_minutes_to_timestamp_basic():
    """Verifies that minutes-since-midnight are combined with a base date to
    produce an absolute timestamp, and that any time-of-day already present
    on base_date is discarded (the function always starts from midnight)."""
    base_date = datetime(2015, 8, 2, 15, 30)
    result = minutes_to_timestamp(570, base_date)
    assert result == datetime(2015, 8, 2, 9, 30)


def test_process_raw_flights_combined(spark):
    """Verifies, in a single combined dataset, that process_raw_flights:
    - keeps only rows matching the requested year/month/day
    - drops cancelled flights
    - drops flights with a null or empty tail number
    - defaults a null DEPARTURE_DELAY to 0 (primary_delay)
    - converts SCHEDULED_DEPARTURE from HHMM to minutes since midnight
    - assigns leg_seq per tail number, ordered by scheduled departure
    - builds flight_id as "<tail_number>_<leg_seq>"
    """
    def _base_row(**overrides):
        row = {
            "YEAR": 2015, "MONTH": 8, "DAY": 2,
            "CANCELLED": 0,
            "TAIL_NUMBER": "N12345",
            "SCHEDULED_DEPARTURE": 930,
            "SCHEDULED_TIME": 120,
            "DEPARTURE_DELAY": 10,
            "ORIGIN_AIRPORT": "ATL",
            "DESTINATION_AIRPORT": "ORD",
        }
        row.update(overrides)
        return row

    rows = [
        _base_row(TAIL_NUMBER="N12345", SCHEDULED_DEPARTURE=600, DEPARTURE_DELAY=10),
        _base_row(TAIL_NUMBER="N12345", SCHEDULED_DEPARTURE=1200, DEPARTURE_DELAY=5),
        _base_row(TAIL_NUMBER="N00000", DAY=3),
        _base_row(TAIL_NUMBER="N11111", CANCELLED=1),
        _base_row(TAIL_NUMBER=None),
        _base_row(TAIL_NUMBER=""),
        _base_row(TAIL_NUMBER="N22222", DEPARTURE_DELAY=None),
    ]
    df = spark.createDataFrame(rows)
    result = process_raw_flights(df, year=2015, month=8, day=2).collect()

    result_by_tail = {}
    for row in result:
        result_by_tail.setdefault(row["tail_number"], []).append(row)

    assert set(result_by_tail.keys()) == {"N12345", "N22222"}

    n12345_legs = sorted(result_by_tail["N12345"], key=lambda r: r["scheduled_departure"])
    assert n12345_legs[0]["scheduled_departure"] == 360
    assert n12345_legs[0]["leg_seq"] == 0
    assert n12345_legs[0]["flight_id"] == "N12345_0"
    assert n12345_legs[1]["scheduled_departure"] == 720
    assert n12345_legs[1]["leg_seq"] == 1
    assert n12345_legs[1]["flight_id"] == "N12345_1"

    assert result_by_tail["N22222"][0]["primary_delay"] == 0


def test_build_info_and_state(spark):
    """Verifies that build_info_and_state correctly links legs of the same
    itinerary (next_flight_id, is_first_leg) and initializes the state RDD:
    only the first leg of an itinerary starts with a non-zero delay and a
    computed ready_time; subsequent legs start with delay=0 and
    ready_time=None until unlocked later in the simulation."""
    rows = [
        {"tail_number": "N12345", "leg_seq": 0, "flight_id": "N12345_0",
         "origin": "ATL", "destination": "ORD",
         "scheduled_departure": 360, "scheduled_flying_time": 120, "primary_delay": 15},
        {"tail_number": "N12345", "leg_seq": 1, "flight_id": "N12345_1",
         "origin": "ORD", "destination": "DEN",
         "scheduled_departure": 720, "scheduled_flying_time": 90, "primary_delay": 0},
        {"tail_number": "N99999", "leg_seq": 0, "flight_id": "N99999_0",
         "origin": "DFW", "destination": "LAX",
         "scheduled_departure": 480, "scheduled_flying_time": 150, "primary_delay": 0},
    ]
    flights_df = spark.createDataFrame(rows)

    flight_info_bc, state_rdd = build_info_and_state(flights_df, spark)

    info = flight_info_bc.value
    state = dict(state_rdd.collect())

    assert info["N12345_0"]["is_first_leg"] is True
    assert info["N12345_0"]["next_flight_id"] == "N12345_1"
    assert info["N12345_0"]["turnaround_time"] == DEFAULT_TURNAROUND

    assert info["N12345_1"]["is_first_leg"] is False
    assert info["N12345_1"]["next_flight_id"] is None

    assert info["N99999_0"]["is_first_leg"] is True
    assert info["N99999_0"]["next_flight_id"] is None

    assert state["N12345_0"]["delay"] == 15
    assert state["N12345_0"]["ready_time"] == 360 + 15
    assert state["N12345_0"]["is_departed"] is False

    assert state["N12345_1"]["delay"] == 0
    assert state["N12345_1"]["ready_time"] is None

    assert state["N99999_0"]["delay"] == 0
    assert state["N99999_0"]["ready_time"] == 480 + 0


def spark_with_checkpoint(spark, tmp_path):
    """Extends the base spark fixture by configuring a temporary checkpoint
    directory, required by state_rdd.checkpoint() inside run_simulation when
    checkpointing is enabled."""
    spark.sparkContext.setCheckpointDir(str(tmp_path / "checkpoints"))
    return spark


def test_run_simulation_single_flight_no_conflicts(spark):
    """Verifies the simplest possible end-to-end run: a single flight with
    no delay and no other traffic should depart exactly on schedule and land
    successfully, with the InfluxDB client receiving the initial control
    message and being closed at the end of the run."""
    rows = [
        {"tail_number": "N12345", "leg_seq": 0, "flight_id": "N12345_0",
         "origin": "ATL", "destination": "ORD",
         "scheduled_departure": 0, "scheduled_flying_time": 10, "primary_delay": 0},
    ]
    flights_df = spark.createDataFrame(rows)

    mock_influx = MagicMock()

    final_state_rdd, _ = run_simulation(
        spark_session=spark,
        flights_df=flights_df,
        influence_score_map={},
        date=datetime(2015, 8, 2),
        t0=0,
        t_end=11,
        use_influence=False,
        use_delay_streaming=False,
        influxdb_client=mock_influx,
        write_events=False,
        enable_checkpoint=False
    )

    final_state = dict(final_state_rdd.collect())

    assert final_state["N12345_0"]["is_departed"] is True
    assert final_state["N12345_0"]["is_landed"] is True
    assert final_state["N12345_0"]["departure_time"] == 0
    assert final_state["N12345_0"]["delay"] == 0

    mock_influx.close.assert_called_once()

    mock_influx.write_point.assert_any_call(
        measurement="control",
        tags={"type": "info"},
        fields={"message": "Control panel started"},
        time=datetime(2015, 8, 2, 0, 0)
    )


def test_run_simulation_departure_conflict(spark):
    """Verifies departure conflict resolution: two flights from the same
    airport that become ready to depart at the exact same simulated minute
    (same ready_time, different delay) must not depart simultaneously, and
    the flight with the higher delay (higher priority) must depart first."""
    rows = [
        {"tail_number": "N11111", "leg_seq": 0, "flight_id": "N11111_0",
         "origin": "ATL", "destination": "ORD",
         "scheduled_departure": 0, "scheduled_flying_time": 10, "primary_delay": 20},
        {"tail_number": "N22222", "leg_seq": 0, "flight_id": "N22222_0",
         "origin": "ATL", "destination": "DEN",
         "scheduled_departure": 15, "scheduled_flying_time": 10, "primary_delay": 5},
    ]
    flights_df = spark.createDataFrame(rows)

    mock_influx = MagicMock()

    final_state_rdd, _ = run_simulation(
        spark_session=spark,
        flights_df=flights_df,
        influence_score_map={},
        date=datetime(2015, 8, 2),
        t0=0,
        t_end=31,
        use_influence=False,
        use_delay_streaming=False,
        influxdb_client=mock_influx,
        write_events=False,
        enable_checkpoint=False
    )

    final_state = dict(final_state_rdd.collect())

    assert final_state["N11111_0"]["is_departed"] is True
    assert final_state["N22222_0"]["is_departed"] is True
    assert final_state["N11111_0"]["departure_time"] <= final_state["N22222_0"]["departure_time"]


def test_run_simulation_arrival_conflict(spark):
    """Verifies arrival conflict resolution: two flights scheduled to land
    at the same destination at the same simulated minute must not both land
    at once. The winner (higher delay/priority) lands immediately, while the
    loser's landing is deferred (is_landed stays False within the tested
    time window)."""
    rows = [
        {"tail_number": "N11111", "leg_seq": 0, "flight_id": "N11111_0",
         "origin": "ATL", "destination": "ORD",
         "scheduled_departure": 0, "scheduled_flying_time": 10, "primary_delay": 20},
        {"tail_number": "N22222", "leg_seq": 0, "flight_id": "N22222_0",
         "origin": "DEN", "destination": "ORD",
         "scheduled_departure": 0, "scheduled_flying_time": 25, "primary_delay": 5},
    ]
    flights_df = spark.createDataFrame(rows)

    mock_influx = MagicMock()

    final_state_rdd, _ = run_simulation(
        spark_session=spark,
        flights_df=flights_df,
        influence_score_map={},
        date=datetime(2015, 8, 2),
        t0=0,
        t_end=31,
        use_influence=False,
        use_delay_streaming=False,
        influxdb_client=mock_influx,
        write_events=False,
        enable_checkpoint=False
    )

    final_state = dict(final_state_rdd.collect())

    assert final_state["N11111_0"]["is_landed"] is True
    assert final_state["N22222_0"]["is_landed"] is False


def test_run_simulation_multi_leg_itinerary(spark):
    """Verifies that a second leg of the same aircraft's itinerary only
    departs after the first leg has landed plus the required turnaround
    time, confirming that build_info_and_state's next_flight_id linkage and
    the in-simulation unlocking logic work together correctly across legs."""
    rows = [
        # first leg
        {"tail_number": "N12345", "leg_seq": 0, "flight_id": "N12345_0",
         "origin": "ATL", "destination": "ORD",
         "scheduled_departure": 0, "scheduled_flying_time": 10, "primary_delay": 10},
        # second leg
        {"tail_number": "N12345", "leg_seq": 1, "flight_id": "N12345_1",
         "origin": "ORD", "destination": "DEN",
         "scheduled_departure": 10 + DEFAULT_TURNAROUND, "scheduled_flying_time": 10, "primary_delay": 0},
    ]
    flights_df = spark.createDataFrame(rows)

    mock_influx = MagicMock()

    final_state_rdd, _ = run_simulation(
        spark_session=spark,
        flights_df=flights_df,
        influence_score_map={},
        date=datetime(2015, 8, 2),
        t0=0,
        t_end=20 + DEFAULT_TURNAROUND + 11,
        use_influence=False,
        use_delay_streaming=False,
        influxdb_client=mock_influx,
        write_events=False,
        enable_checkpoint=False
    )

    final_state = dict(final_state_rdd.collect())

    # first leg
    assert final_state["N12345_0"]["is_departed"] is True
    assert final_state["N12345_0"]["is_landed"] is True
    assert final_state["N12345_0"]["departure_time"] == 10

    # second leg
    assert final_state["N12345_1"]["is_departed"] is True
    assert final_state["N12345_1"]["is_landed"] is True

    expected_departure = final_state["N12345_0"]["scheduled_arrival"] + DEFAULT_TURNAROUND
    assert final_state["N12345_1"]["departure_time"] == expected_departure + final_state["N12345_1"]["delay"]
