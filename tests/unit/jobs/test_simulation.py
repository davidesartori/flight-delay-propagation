"""Testing module for simulation job"""
from unittest.mock import MagicMock
from datetime import datetime
from zoneinfo import ZoneInfo
from src.jobs.simulation import (convert_minutes_to_hhmm,
                                 convert_to_central_minutes,
                                 minutes_to_timestamp,
                                 process_raw_flights,
                                 build_info_and_state,
                                 run_simulation,
                                 DEFAULT_TURNAROUND,
                                 FLIGHT_SEPARATION)


class TestConvertToCentralMinutes:
    """Tests for conversion of local HHMM to minutes since Central midnight."""

    def test_new_york_to_central(self):
        """00:30 EDT is 23:30 CDT -> -45 minutes."""
        assert convert_to_central_minutes(
            30, "JFK", {"JFK": "America/New_York"}, 2015, 7, 1) == -30

    def test_los_angeles_to_central(self):
        """14:30 PDT is 16:30 CDT -> 990 minutes."""
        assert convert_to_central_minutes(
            1430, "LAX", {"LAX": "America/Los_Angeles"}, 2015, 7, 1) == 16 * 60 + 30

    def test_already_central(self):
        """An airport already in Central Time is unchanged."""
        assert convert_to_central_minutes(
            1430, "ORD", {"ORD": "America/Chicago"}, 2015, 7, 1) == 14 * 60 + 30

    def test_none_time(self):
        """A missing time returns None."""
        assert convert_to_central_minutes(
            None, "JFK", {"JFK": "America/New_York"}, 2015, 7, 1) is None

    def test_unknown_airport_defaults_to_central(self):
        """An airport missing from the tz map is treated as Central."""
        assert convert_to_central_minutes(
            1430, "UNKNOWN", {"JFK": "America/New_York"}, 2015, 7, 1) == 14 * 60 + 30

    def test_short_hhmm_without_leading_zero(self):
        """HHMM=30 (00:30 EDT) is 23:30 CDT of the previous day -> -30."""
        assert convert_to_central_minutes(
            30, "JFK", {"JFK": "America/New_York"}, 2015, 7, 1) == -30

    def test_next_day_overflow(self):
        """23:30 PDT is 01:30 CDT of the next day -> 1530."""
        assert convert_to_central_minutes(
            2330, "LAX", {"LAX": "America/Los_Angeles"}, 2015, 7, 1) == 1440 + 90

    def test_2400_is_next_midnight(self):
        """HHMM=2400 at a Central airport is exactly 1440."""
        assert convert_to_central_minutes(
            2400, "ORD", {"ORD": "America/Chicago"}, 2015, 7, 1) == 1440

    def test_daylight_saving_time(self):
        """In July, 12:00 EDT is 11:00 CDT (offset is 1h)."""
        assert convert_to_central_minutes(
            1200, "JFK", {"JFK": "America/New_York"}, 2015, 7, 1) == 11 * 60


def test_convert_minutes_to_hhmm_basic():
    """Verifies that minutes-since-midnight are formatted back into a
    zero-padded HH:MM string."""
    assert convert_minutes_to_hhmm(570) == "09:30"


def test_minutes_to_timestamp_basic():
    """Verifica che i minuti siano applicati alla mezzanotte di Chicago."""
    chicago = ZoneInfo("America/Chicago")
    base_date = datetime(2015, 8, 2, 15, 30, tzinfo=chicago)

    result = minutes_to_timestamp(570, base_date)

    assert result == datetime(2015, 8, 2, 9, 30, tzinfo=chicago)


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
        _base_row(TAIL_NUMBER="N12345", SCHEDULED_DEPARTURE=1200, DEPARTURE_DELAY=5, ORIGIN_AIRPORT="ORD"),
        _base_row(TAIL_NUMBER="N00000", DAY=3),
        _base_row(TAIL_NUMBER="N11111", CANCELLED=1),
        _base_row(TAIL_NUMBER=None),
        _base_row(TAIL_NUMBER=""),
        _base_row(TAIL_NUMBER="N22222", DEPARTURE_DELAY=None),
    ]
    df = spark.createDataFrame(rows)
    result = process_raw_flights(spark, df, year=2015, month=8, day=2).collect()

    result_by_tail = {}
    for row in result:
        result_by_tail.setdefault(row["tail_number"], []).append(row)

    assert set(result_by_tail.keys()) == {"N12345", "N22222"}

    n12345_legs = sorted(result_by_tail["N12345"], key=lambda r: r["scheduled_departure"])
    assert n12345_legs[0]["scheduled_departure"] == 300
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
    successfully"""
    rows = [
        {"tail_number": "N12345", "leg_seq": 0, "flight_id": "N12345_0",
         "origin": "ATL", "destination": "ORD",
         "scheduled_departure": 0, "scheduled_flying_time": 10, "primary_delay": 0},
    ]
    flights_df = spark.createDataFrame(rows)

    mock_influx = MagicMock()

    final_state_rdd, _, _ = run_simulation(
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

    final_state_rdd, _, _ = run_simulation(
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

    final_state_rdd, _, _ = run_simulation(
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

    final_state_rdd, _, _ = run_simulation(
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


def test_run_simulation_streaming_delay_on_airport(spark, monkeypatch):
    """Verifies that a delay received from the stream for an airport is
    applied to the flights waiting to depart from it: two flights from ATL
    are scheduled early (minutes 10 and 12), a 20-minute delay for ATL
    arrives at minute 5, and both flights must depart only after the delayed
    time (minute 5 + 20), not at their original schedule.
 
    Flights are eligible to depart up to `window` minutes before
    scheduled_departure + delay (see departing_candidates in the job), so
    the earliest allowed departure is injection_time + delay - window."""
    injection_time = 5
    injected_delay = 20
    window = min(FLIGHT_SEPARATION - 1, 5)
 
    rows = [
        {"tail_number": "N11111", "leg_seq": 0, "flight_id": "N11111_0",
         "origin": "ATL", "destination": "ORD",
         "scheduled_departure": 10, "scheduled_flying_time": 10, "primary_delay": 0},
        {"tail_number": "N22222", "leg_seq": 0, "flight_id": "N22222_0",
         "origin": "ATL", "destination": "DEN",
         "scheduled_departure": 12, "scheduled_flying_time": 10, "primary_delay": 0},
    ]
    flights_df = spark.createDataFrame(rows)
 
    # No real generator thread and no real Spark stream
    monkeypatch.setattr("src.jobs.simulation.generate_random_delay_files",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr("src.jobs.simulation.start_delay_stream",
                        lambda *args, **kwargs: (MagicMock(), None, None))
 
    # consume_pending_delays is called once per simulated minute, starting
    # from t0: the call with index `injection_time` corresponds to that minute
    calls = {"n": 0}
 
    def fake_consume_pending_delays(_buffer, _lock):
        n = calls["n"]
        calls["n"] += 1
        return {"ATL": injected_delay} if n == injection_time else {}
 
    monkeypatch.setattr("src.jobs.simulation.consume_pending_delays",
                        fake_consume_pending_delays)
 
    final_state_rdd, _, _ = run_simulation(
        spark_session=spark,
        flights_df=flights_df,
        influence_score_map={},
        date=datetime(2015, 8, 2),
        t0=0,
        t_end=60,
        use_influence=False,
        use_delay_streaming=True,
        influxdb_client=MagicMock(),
        write_events=False,
        enable_checkpoint=False
    )
 
    final_state = dict(final_state_rdd.collect())
 
    earliest_allowed = injection_time + injected_delay - window
 
    assert final_state["N11111_0"]["is_departed"] is True
    assert final_state["N22222_0"]["is_departed"] is True
 
    # both flights are held back until the airport delay has elapsed
    assert final_state["N11111_0"]["departure_time"] == 23
    assert final_state["N22222_0"]["departure_time"] == 23 + FLIGHT_SEPARATION
 
    # they still do not depart at the same minute
    assert (final_state["N11111_0"]["departure_time"]
            != final_state["N22222_0"]["departure_time"])
