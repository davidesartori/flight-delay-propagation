"""Flight delay propagation simulation."""

import logging
import argparse
import yaml
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark import StorageLevel
from src.utils.io import load__influence_scores
from src.utils.spark import create_spark_session
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

spark_session = create_spark_session(app_name="Simulation")
spark_session.sparkContext.setCheckpointDir("checkpoint_dir")

FLIGHT_SEPARATION = 2  # minutes between takeoffs at same airport
DEFAULT_TURNAROUND = 45  # plane turnaround time
CHECKPOINT_EVERY = 60
T_START = 0
T_MAX = 2880  # 2 days in minutes
INFLUENCE_RATE = 5  # influence factor for delay propagation


def parse_hhmm_to_minutes(col):
    """Parse HHMM time format to minutes since midnight."""
    s = F.lpad(col.cast("string"), 4, "0")
    hh = F.substring(s, 1, 2).cast("int")
    mm = F.substring(s, 3, 2).cast("int")
    return (hh % 24) * 60 + mm


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


def build_info_and_state(flights_df):
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
            "delay": r.primary_delay if is_first else 0,
            "ready_time": r.scheduled_departure + r.primary_delay if is_first else None,
            "is_departed": False,
        }))

    flight_info_bc = spark_session.sparkContext.broadcast(flight_info)
    state_rdd = spark_session.sparkContext.parallelize(flight_state)

    return flight_info_bc, state_rdd


def get_magnitude(flight_info_bc, state_rdd, t):
    """Calculate the magnitude of delays at each airport."""
    state_rdd = state_rdd.filter(
        lambda flight: flight[1]["is_departed"] is True and flight[1]["departure_time"] < t and
        flight[1]["departure_time"] >= t-60)
    delay_time_per_airport = state_rdd.map(lambda flight:
                                           (flight_info_bc.value[flight[0]]["origin"],
                                            max(flight[1]["departure_time"] -
                                                flight[1]["scheduled_departure"],
                                                0))).reduceByKey(lambda a, b:
                                                                 (a + b)).filter(lambda item:
                                                                                 item[1] != 0)

    delay_count_per_airport = state_rdd.map(lambda flight: (flight_info_bc.value[flight[0]]["origin"],
                                                            1 if flight[1]["departure_time"] -
                                                            flight[1]["scheduled_departure"] >= 1 else
                                                            0)).reduceByKey(lambda a, b:
                                                                            (a + b)).filter(lambda item: item[1] != 0)

    return delay_count_per_airport, delay_time_per_airport


def run_simulation(flights_df, influence_score_map, t0=0, t_end=1440, use_influence=True):
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

    def resolve_landing_group(item, t):
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

        for flight_id, _ in flights:
            if flight_id != winner_id:
                info = flight_info_bc.value[flight_id]

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

        return state

    flight_info_bc, state_rdd = build_info_and_state(flights_df)

    influence_bc = None

    if use_influence:
        influence_bc = spark_session.sparkContext.broadcast(
            influence_score_map)

    state_rdd = state_rdd.persist(StorageLevel.MEMORY_AND_DISK)

    not_departed = state_rdd.filter(
        lambda flight: flight[1]["is_departed"] is False)

    t = t0
    while not_departed.count() > 0 and t < t_end:
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

        landing_conflicts = state_rdd.filter(lambda flight: flight[1]["is_departed"] is True and
                                             t == flight[1]["departure_time"] + flight_info_bc.value[flight[0]]["scheduled_flying_time"])

        landing_conflicts = landing_conflicts.map(lambda conflict: (
            flight_info_bc.value[conflict[0]]["destination"], conflict)).groupByKey()

        resolved_landing = landing_conflicts.flatMap(
            lambda item: resolve_landing_group(item, t))

        state_rdd = state_rdd.union(resolved).union(
            resolved_landing).reduceByKey(merge_states, numPartitions=10)

        if (t - t0) % CHECKPOINT_EVERY == 0:
            state_rdd = state_rdd.persist(StorageLevel.MEMORY_AND_DISK)
            state_rdd.checkpoint()
            state_rdd.count()

        if (t - t0) % 60 == 0 and t > t0:
            dp_mag1, dp_mag2 = get_magnitude(flight_info_bc, state_rdd, t)
            # TODO: save data to database

        not_departed = state_rdd.filter(
            lambda flight: flight[1]["is_departed"] is False)
        t += 1

    return state_rdd


def main():
    """Main function to run the flight delay propagation simulation."""
    setup_logging()

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

    final_state = run_simulation(
        flights_df, influence_scores, t0=T_START, t_end=T_MAX, use_influence=True)

    remaining_flights = final_state.filter(
        lambda x: x[1]["is_departed"] is False)

    logger.info("Simulation completed. Remaining flights: %s", remaining_flights.count())


if __name__ == "__main__":
    main()
