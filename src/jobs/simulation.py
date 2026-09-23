"""Flight delay propagation simulation."""

import threading
import logging
import argparse
from datetime import datetime, timedelta
import yaml
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark import StorageLevel
from src.utils.io import load__influence_scores, save_output
from src.utils.spark import create_spark_session
from src.utils.logging import setup_logging
from src.utils.influxdb import InfluxDBService
from src.simulation.delay_generator import generate_random_delay_files
from src.simulation.delay_stream import start_delay_stream, consume_pending_delays
from src.simulation.simulation_stats import get_magnitude, get_speed, compute_simulation_stats

logger = logging.getLogger(__name__)

FLIGHT_SEPARATION = 5  # minutes between takeoffs at same airport
DEFAULT_TURNAROUND = 45  # plane turnaround time
CHECKPOINT_EVERY = 60
T_START = 0
T_MAX = 2880  # 2 days in minutes
INFLUENCE_RATE = 3  # influence factor for delay propagation


def parse_hhmm_to_minutes(col):
    """Parse HHMM time format to minutes since midnight."""
    s = F.lpad(col.cast("string"), 4, "0")
    hh = F.substring(s, 1, 2).cast("int")
    mm = F.substring(s, 3, 2).cast("int")
    return (hh % 24) * 60 + mm


def convert_minutes_to_hhmm(minutes: int):
    """Convert minutes since midnight to HHMM time format."""
    hh = (minutes // 60) % 24
    mm = minutes % 60
    return f"{hh:02d}:{mm:02d}"


def minutes_to_timestamp(minutes: int, base_date: datetime, seconds=0) -> datetime:
    """Convert minutes since midnight to a timestamp on the given base date."""
    return base_date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=minutes, seconds=seconds)


def process_raw_flights(raw_df, year, month, day):
    """Process raw flight data for a specific date, filtering out 
    cancelled flights and those without tail numbers. It also assigns a
    sequence number to each leg of the flight based on the tail number"""
    day_df = raw_df.filter(
        (F.col("YEAR") == year) & (F.col("MONTH") == month) & (F.col("DAY") == day) &
        (F.col("CANCELLED") == 0)
    ).filter(
        F.col("TAIL_NUMBER").isNotNull() & (F.col("TAIL_NUMBER") != "")
    )

    day_df = day_df.withColumn(
        "scheduled_departure", parse_hhmm_to_minutes(
            F.col("SCHEDULED_DEPARTURE"))
    ).withColumn(
        "scheduled_flying_time", F.col("SCHEDULED_TIME").cast("int")
    ).withColumn(
        "primary_delay", F.coalesce(
            F.col("DEPARTURE_DELAY").cast("int"), F.lit(0))
    )

    w = Window.partitionBy("TAIL_NUMBER").orderBy("scheduled_departure")
    day_df = day_df.withColumn("leg_seq", F.row_number().over(w) - 1)

    day_df = day_df.withColumn(
        "flight_id", F.concat_ws("_", F.col(
            "TAIL_NUMBER"), F.col("leg_seq").cast("string"))
    )

    return day_df.select(
        "flight_id",
        F.col("TAIL_NUMBER").alias("tail_number"),
        F.col("ORIGIN_AIRPORT").alias("origin"),
        F.col("DESTINATION_AIRPORT").alias("destination"),
        "scheduled_departure", "scheduled_flying_time", "primary_delay", "leg_seq",
    )


def build_info_and_state(flights_df, spark_session):
    """Builds flight information and initial state RDD from the processed flights DataFrame."""
    rows = flights_df.orderBy("tail_number", "leg_seq").collect()

    itinerary = {}
    for r in rows:
        itinerary.setdefault(r.tail_number, []).append(r.flight_id)

    flight_info, flight_state = {}, []
    for r in rows:
        legs = itinerary[r.tail_number]
        idx = legs.index(r.flight_id)
        next_id = legs[idx + 1] if idx + 1 < len(legs) else None
        is_first = idx == 0

        flight_info[r.flight_id] = {
            "tail_number": r.tail_number,
            "origin": r.origin,
            "destination": r.destination,
            "scheduled_departure": r.scheduled_departure,
            "scheduled_flying_time": r.scheduled_flying_time,
            "turnaround_time": DEFAULT_TURNAROUND,
            "next_flight_id": next_id,
            "is_first_leg": is_first,
        }

        flight_state.append((r.flight_id, {
            "scheduled_departure": r.scheduled_departure,
            "delay": r.primary_delay if is_first and r.primary_delay > 0 else 0,
            "ready_time": r.scheduled_departure + r.primary_delay if is_first else None,
            "is_departed": False,
        }))

    flight_info_bc = spark_session.sparkContext.broadcast(flight_info)
    state_rdd = spark_session.sparkContext.parallelize(flight_state)

    return flight_info_bc, state_rdd


def run_simulation(spark_session, flights_df, influence_score_map, date, t0=0, t_end=1440, use_influence=False, use_delay_streaming=False,
                   delay_streaming_dir="delay_streaming", delay_interval_sec=10, delay_probability=0.001, delay_range=(5, 30),
                   influxdb_client:InfluxDBService=None, write_events=True, enable_checkpoint=True):
    """Run the flight delay propagation simulation."""
    def resolve_group(item, t):
        _, flights = item
        flights = list(flights)

        def priority_key(fs):
            flight_id, state = fs
            destination = flight_info_bc.value[flight_id]["destination"]
            influence = influence_bc.value.get(
                destination, 0) if influence_bc else 0
            delay = state["delay"]
            priority = delay if delay > 0 else 1
            if influence > 0 and delay > 0:
                priority *= (1 + INFLUENCE_RATE * ((influence - 1) / 99)**2)
            return priority

        winner_id, _ = max(flights, key=priority_key)

        updates = []
        for flight_id, state in flights:
            if flight_id == winner_id:
                info = flight_info_bc.value[flight_id]
                new_state = dict(state)
                new_state["is_departed"] = True
                new_state["departure_time"] = t
                new_state["scheduled_arrival"] = t + info["scheduled_flying_time"]
                new_state["is_landed"] = False
                updates.append((flight_id, new_state))

                # unlock next leg if it exists
                if info["next_flight_id"] is not None:
                    next_flight_id = info["next_flight_id"]
                    updates.append((next_flight_id, {"scheduled_departure": 0,
                                                     "delay": 0,
                                                     "is_departed": False,
                                                     "ready_time": t + info["scheduled_flying_time"] + DEFAULT_TURNAROUND}))
            else:
                new_state = dict(state)
                additional_delay = 1 if FLIGHT_SEPARATION <= 6 else 0
                new_state["delay"] = max(
                    0, FLIGHT_SEPARATION + t - (state["scheduled_departure"] + state["delay"])) + additional_delay
                updates.append((flight_id, new_state))

        return updates

    def resolve_landing_group(item):
        _, flights = item
        flights = list(flights)
        updates = []

        def priority_key(fs):
            flight_id, state = fs
            next_flight = flight_info_bc.value[flight_id]["next_flight_id"]

            delay = state["delay"]
            priority = delay if delay > 0 else 1

            if next_flight is not None:
                next_flight_info = flight_info_bc.value[next_flight]
                influence = influence_bc.value.get(next_flight_info["destination"],
                                                   0) if influence_bc else 0
                priority = delay if delay > 0 else 1
                if influence > 0 and delay > 0:
                    priority *= (1 + INFLUENCE_RATE *
                                 ((influence - 1) / 99)**2)

            return priority

        winner_id, _ = max(flights, key=priority_key)

        for flight_id, state in flights:
            if flight_id == winner_id:
                info = flight_info_bc.value[flight_id]
                new_state = dict(state)
                new_state["is_landed"] = True
                updates.append((flight_id, new_state))
            else:
                info = flight_info_bc.value[flight_id]
                new_state = dict(state)
                new_state["scheduled_arrival"] = state["scheduled_arrival"] + FLIGHT_SEPARATION
                updates.append((flight_id, new_state))

                if info["next_flight_id"] is not None:
                    next_flight_id = info["next_flight_id"]

                    updates.append((next_flight_id, {"scheduled_departure": 0,
                                                     "delay": FLIGHT_SEPARATION,
                                                     "is_departed": False,
                                                     "ready_time": None}))

        return updates

    def merge_states(a, b):
        state = dict(a)

        state["scheduled_departure"] = max(a.get("scheduled_departure"),
                                           b.get("scheduled_departure"))
        state["delay"] = (a.get("delay") or 0) + (b.get("delay") or 0)
        state["is_departed"] = a.get("is_departed") or b.get("is_departed")
        state["ready_time"] = max((a.get("ready_time") or 0),
                                  (b.get("ready_time") or 0))
        state["departure_time"] = max((a.get("departure_time") or 0),
                                      (b.get("departure_time") or 0))
        state["scheduled_arrival"] = max((a.get("scheduled_arrival") or 0),
                                      (b.get("scheduled_arrival") or 0))
        state["is_landed"] = a.get("is_landed") or b.get("is_landed")

        return state

    flight_info_bc, state_rdd = build_info_and_state(flights_df, spark_session)

    if use_delay_streaming:
        airports = flights_df.select("origin").distinct().rdd.flatMap(lambda x: x).collect()

        stop_event = threading.Event()
        generator_thread = threading.Thread(
            target=generate_random_delay_files,
            args=(airports, delay_streaming_dir, delay_interval_sec, delay_probability, delay_range),
            kwargs={"stop_event": stop_event},
            daemon=True,
        )
        generator_thread.start()

        stream_query, delay_buffer, buffer_lock = start_delay_stream(spark_session, delay_streaming_dir)

    influence_bc = None

    if use_influence:
        influence_bc = spark_session.sparkContext.broadcast(
            influence_score_map)

    state_rdd = state_rdd.persist(StorageLevel.MEMORY_AND_DISK)

    series_mag1_rdd = None
    series_mag2_rdd = None

    n_total_airports = flights_df.select("origin").distinct().count()

    t = t0

    influxdb_client.write_point(
        measurement="control",
        tags={"type": "info"},
        fields={"message": "Control panel started"},
        time=minutes_to_timestamp(t, date)
    )

    not_departed = state_rdd.filter(
        lambda flight: flight[1]["is_departed"] is False)

    not_landed = state_rdd.filter(
        lambda flight: flight[1].get("is_landed", False) is False)

    # simulation
    while (not_departed.count() > 0 or not_landed.count() > 0) and t < t_end:
        # delays
        if use_delay_streaming:
            pending_delays = consume_pending_delays(delay_buffer, buffer_lock)

            if pending_delays:
                for airport, delay in pending_delays.items():
                    affected_flights = not_departed.filter(
                            lambda flight, airport=airport, delay=delay: flight_info_bc.value[flight[0]]["origin"] == airport and
                            flight[1]["ready_time"] is not None and
                            flight[1]["ready_time"] <= t + delay and
                            flight[1]["scheduled_departure"] + flight[1]["delay"] <= t + delay and
                            flight[1]["scheduled_departure"] + flight[1]["delay"] >= t)

                    if affected_flights.count() > 0:
                        logger.info("Applying delay of %d minutes to %d flights at airport %s at time %s",
                                    delay, affected_flights.count(), airport, convert_minutes_to_hhmm(t))

                        influxdb_client.write_point(
                            measurement="events",
                            tags={},
                            fields={"message": f"Applying delay of {delay} minutes to {affected_flights.count()} flights at airport {airport} at time {convert_minutes_to_hhmm(t)}"},
                            time=minutes_to_timestamp(t, date)
                        )

                        affected_flights = affected_flights.map(lambda flight, delay=delay: (
                                flight[0], {
                                    **flight[1],
                                    "delay": (t + delay) - (flight[1]["scheduled_departure"] + flight[1]["delay"]),
                                }))

                        not_departed = not_departed.subtractByKey(affected_flights).union(affected_flights)

        # departures
        departing_candidates = not_departed.filter(
            lambda flight:
            flight[1]["scheduled_departure"] + flight[1]["delay"] <= t + min(FLIGHT_SEPARATION - 1, 5) and
            flight[1]["ready_time"] is not None and
            flight[1]["ready_time"] <= t
        )
        departing_candidates = departing_candidates.map(lambda candidate: (
            flight_info_bc.value[candidate[0]]["origin"], candidate)).groupByKey()
        resolved = departing_candidates.flatMap(
            lambda item: resolve_group(item, t))

        if influxdb_client and write_events:
            airports_with_conflicts = departing_candidates.filter(lambda flight: len(flight[1]) > 1).keys().collect()
            winner_flights = resolved.filter(lambda flight: flight[1]["is_departed"])
            resolved_flight_ids = winner_flights.filter(lambda flight: flight_info_bc.value[flight[0]]["origin"] in airports_with_conflicts).keys().collect()

            resolutions = [
                {
                    "measurement": "control",
                    "tags": {"type": "resolution"},
                    "fields": {
                        "message": f"Departure conflict: prioritizing flight {resolved_flight_id} at airport {flight_info_bc.value[resolved_flight_id]['origin']}"
                    },
                    "time": minutes_to_timestamp(t, date, 0)
                }
                for resolved_flight_id in resolved_flight_ids
            ]
            influxdb_client.write_points(resolutions)

            if not winner_flights.isEmpty():
                departed_flights_messages_rdd = winner_flights.map(lambda flight: (f"{flight_info_bc.value[flight[0]]['origin']} -> {flight_info_bc.value[flight[0]]['destination']} | Departed at {convert_minutes_to_hhmm(t)} with delay of {flight[1]["delay"]} minutes", minutes_to_timestamp(t, date)))
                departed_flights_messages_df = spark_session.createDataFrame(departed_flights_messages_rdd, schema=["message", "timestamp"])

                influxdb_client.write_dataframe(departed_flights_messages_df, measurement="events", tag_cols=[], field_cols=["message"], time_col="timestamp")

        # arrivals
        landing_conflicts = state_rdd.filter(lambda flight: flight[1]["is_departed"] is True and
                                             t == flight[1]["scheduled_arrival"])

        landing_conflicts = landing_conflicts.map(lambda conflict: (
            flight_info_bc.value[conflict[0]]["destination"], conflict)).groupByKey()

        resolved_landing = landing_conflicts.flatMap(resolve_landing_group)

        if influxdb_client and write_events:
            airports_with_conflicts = landing_conflicts.filter(lambda flight: len(flight[1]) > 1).keys().collect()
            winner_flights = resolved_landing.filter(lambda flight: flight[1].get("is_landed", False))
            resolved_flight_ids = winner_flights.filter(lambda flight: flight_info_bc.value[flight[0]]["destination"] in airports_with_conflicts).keys().collect()

            resolutions = [
                {
                    "measurement": "control",
                    "tags": {"type": "resolution"},
                    "fields": {
                        "message": f"Arrival conflict: prioritizing flight {resolved_flight_id} at airport {flight_info_bc.value[resolved_flight_id]['destination']}"
                    },
                    "time": minutes_to_timestamp(t, date, 0)
                }
                for resolved_flight_id in resolved_flight_ids
            ]
            influxdb_client.write_points(resolutions)

        state_rdd = state_rdd.union(resolved).union(
            resolved_landing).reduceByKey(merge_states, numPartitions=10)

        if enable_checkpoint and (t - t0) % CHECKPOINT_EVERY == 0:
            state_rdd = state_rdd.persist(StorageLevel.MEMORY_AND_DISK)
            state_rdd.checkpoint()
            state_rdd.count()

        if (t - t0) % 60 == 0 and t > t0:
            dp_mag1, dp_mag2 = get_magnitude(flight_info_bc, state_rdd, t)
            dp_spe = get_speed(dp_mag1, n_total_airports)

            influxdb_client.write_point(
                measurement="dp_spe",
                tags={},
                fields={"speed": dp_spe},
                time=minutes_to_timestamp(t, date)
            )

            magnitude1_at_t = dp_mag1.mapValues(
                lambda magnitude, t=t: (t, magnitude)
            )

            series_mag1_rdd = (
                magnitude1_at_t
                if series_mag1_rdd is None
                else series_mag1_rdd.union(magnitude1_at_t)
            )

            magnitude2_at_t = dp_mag2.mapValues(
                lambda magnitude, t=t: (t, magnitude)
            )

            series_mag2_rdd = (
                magnitude2_at_t
                if series_mag2_rdd is None
                else series_mag2_rdd.union(magnitude2_at_t)
            )

            dp_mag1 = dp_mag1.map(lambda x: (minutes_to_timestamp(t, date), x[0], x[1]))
            dp_mag2 = dp_mag2.map(lambda x: (minutes_to_timestamp(t, date), x[0], x[1]))

            if(not dp_mag1.isEmpty() and not dp_mag2.isEmpty()):
                dp_mag1 = spark_session.createDataFrame(dp_mag1, schema=["timestamp", "origin", "delay_count"])
                dp_mag2 = spark_session.createDataFrame(dp_mag2, schema=["timestamp", "origin", "delay_time"])
                influxdb_client.write_dataframe(dp_mag1, measurement="dp_mag1", tag_cols=["origin"], field_cols=["delay_count"], time_col="timestamp")
                influxdb_client.write_dataframe(dp_mag2, measurement="dp_mag2", tag_cols=["origin"], field_cols=["delay_time"], time_col="timestamp")

        not_departed = state_rdd.filter(
            lambda flight: flight[1]["is_departed"] is False)

        not_landed = state_rdd.filter(
            lambda flight: flight[1].get("is_landed", False) is False)
        
        t += 1

    influxdb_client.close()

    if use_delay_streaming:
        stop_event.set()
        stream_query.stop()

    simulation_stats = compute_simulation_stats(state_rdd, flight_info_bc, influence_score_map)

    return state_rdd, simulation_stats


def save_stats(path, stats, label):
    """Save stats to file"""
    output = (
        f"=== System Delay Statistics {label} ===\n"
        f"Departed flights:                    {stats['n_departed_flights']}\n"
        f"Flights not departed (t_end):        {stats['n_not_departed']}\n"
        f"Delayed flights (>=1 min):            {stats['n_delayed_flights']}\n"
        f"Delayed flights (>=60 min):           {stats['n_delays_gt_60']}\n"
        f"Delayed flights (>=120 min):          {stats['n_delays_gt_120']}\n"
        f"Delayed flights (>=200 min):          {stats['n_delays_gt_200']}\n"
        f"Delayed flights (>=250 min):          {stats['n_delays_gt_250']}\n"
        f"Delayed flights (>=300 min):          {stats['n_delays_gt_300']}\n"
        f"Delayed flights (>=400 min):          {stats['n_delays_gt_400']}\n"
        f"Total delay:                          {stats['total_delay']:.1f} minutes\n"
        f"Average delay (all flights):          {stats['avg_delay']:.2f} minutes\n"
        f"Average delay (delayed flights):      {stats['avg_delay_delayed']:.2f} minutes\n"
        f"Median delay:                         {stats['median_delay']:.1f} minutes\n"
        f"Percentiles:                          {stats['percentiles']}\n"
        f"Maximum delay:                        {stats['max_delay']:.1f} minutes\n"
        f"--- Flights to top-N airports by influence ---\n"
        f"Number of flights:                    {stats['n_top_airport_flights']}\n"
        f"Average delay:                        {stats['top_avg_delay']:.2f} minutes\n"
        f"Median delay:                         {stats['top_median_delay']:.1f} minutes\n"
        f"Total delay:                          {stats['top_total_delay']:.1f} minutes\n"
        f"Influence-weighted total delay:       {stats['total_weighted_delay']:.1f}"
    )

    save_output(path, output)


def main():
    """Main function to run the flight delay propagation simulation."""
    setup_logging()

    spark_session = create_spark_session(app_name="Simulation")
    spark_session.sparkContext.setCheckpointDir("checkpoint_dir")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Loading influence scores")

    influence_scores = load__influence_scores(config["input"]["influence"])

    raw_flights_df = spark_session.read.csv(
        config["input"]["flights"], header=True, inferSchema=True)

    flights_df = process_raw_flights(raw_flights_df, year=2015, month=8, day=2)
    flights_df = flights_df.persist()

    logger.info("Starting simulation")

    day = config["simulation"].get("day")
    month = config["simulation"].get("month")
    year = config["simulation"].get("year")
    use_influence = config["simulation"].get("influence", True)
    use_delay_streaming = config["simulation"].get("delay_streaming", False)
    delay_streaming_dir = config["delay_streaming"].get("input_dir", "delay_streaming")
    delay_interval_sec = config["delay_streaming"].get("interval_sec", 10)
    delay_probability = config["delay_streaming"].get("probability", 0.001)
    delay_range = tuple(config["delay_streaming"].get("delay_range", (5, 30)))
    write_events = config["simulation"].get("write_events", True)
    output_path = config["output"].get("simulation", "output/sim_result.txt")

    influxdb_client = InfluxDBService()

    influxdb_client.delete_table(table="dp_mag1", hard=True)
    influxdb_client.delete_table(table="dp_mag2", hard=True)
    influxdb_client.delete_table(table="dp_spe", hard=True)
    influxdb_client.delete_table(table="events", hard=True)
    influxdb_client.delete_table(table="control", hard=True)

    final_state, simulation_stats = run_simulation(spark_session,
        flights_df, influence_scores, date=datetime(year, month, day), t0=T_START,
        t_end=T_MAX, use_influence=use_influence,
        use_delay_streaming=use_delay_streaming, delay_streaming_dir=delay_streaming_dir,
        delay_interval_sec=delay_interval_sec, delay_probability=delay_probability,
        delay_range=delay_range, influxdb_client=influxdb_client, write_events=write_events, enable_checkpoint=True)

    remaining_flights = final_state.filter(
        lambda x: x[1]["is_departed"] is False)

    logger.info("Simulation completed. Remaining flights: %s", remaining_flights.count())

    save_stats(output_path, simulation_stats, label="")


if __name__ == "__main__":
    main()
