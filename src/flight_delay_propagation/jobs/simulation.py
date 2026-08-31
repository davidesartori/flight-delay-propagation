from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark import SparkContext, StorageLevel

spark = (
    SparkSession.builder
    .appName("Simulator")
    .getOrCreate()
)

sc = SparkContext.getOrCreate()
sc.setCheckpointDir("checkpoint_dir")

FLIGHT_SEPARATION = 2  # minutes between takeoffs at same airport
DEFAULT_TURNAROUND = 45  # plane turnaround time
CHECKPOINT_EVERY = 60


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

    flight_info_bc = sc.broadcast(flight_info)
    state_rdd = sc.parallelize(flight_state)

    return flight_info_bc, state_rdd


def run_simulation(flights_df, influence_score_map, t0=0, t_end=1440, use_influence=True):
    """Run the flight delay propagation simulation."""
    def resolve_group(item, t):
        origin, flights = item
        flights = list(flights)

        if len(flights) > 1:
            print(str(len(flights)) +
                  " flights waiting for departure at " + origin + " airport")

        def priority_key(fs):
            flight_id, state = fs
            # destination = flight_info_bc.value[flight_id]["destination"]
            # influence = influence_bc.value.get(destination, 0) if influence_bc else 0
            delay = state["delay"]
            priority = delay
            return priority

        winner_id, _ = max(flights, key=priority_key)

        updates = []
        for flight_id, state in flights:
            if flight_id == winner_id:
                info = flight_info_bc.value[flight_id]
                new_state = dict(state)
                new_state["is_departed"] = True
                updates.append((flight_id, new_state))
                print("Aircraft " + flight_id.split("_")[0] + " leg " + flight_id.split("_")[1] +
                      " departed from " + info["origin"] + " at time t: " + str(t))

                # unlock next leg if it exists
                if info["next_flight_id"] is not None:
                    next_flight_id = info["next_flight_id"]
                    updates.append((next_flight_id, {"scheduled_departure": 0,
                                                     "delay": 0,
                                                     "is_departed": False,
                                                     "ready_time": t +
                                                     info["scheduled_flying_time"] +
                                                     DEFAULT_TURNAROUND}))
            else:
                new_state = dict(state)
                additional_delay = 1 if FLIGHT_SEPARATION <= 6 else 0
                new_state["delay"] = max(
                    0, FLIGHT_SEPARATION + t - (state["scheduled_departure"] +
                                                state["delay"])) + additional_delay
                updates.append((flight_id, new_state))

        return updates

    def merge_states(a, b):
        state = dict(a)

        state["scheduled_departure"] = max(
            a.get("scheduled_departure"), b.get("scheduled_departure"))
        state["delay"] = (a.get("delay") or 0) + (b.get("delay") or 0)
        state["is_departed"] = a.get("is_departed") or b.get("is_departed")
        state["ready_time"] = max(
            (a.get("ready_time") or 0), (b.get("ready_time") or 0))

        return state

    flight_info_bc, state_rdd = build_info_and_state(flights_df)

    influence_bc = None

    if use_influence:
        influence_bc = sc.broadcast(influence_score_map)

    state_rdd = state_rdd.persist(StorageLevel.MEMORY_AND_DISK)

    not_departed = state_rdd.filter(
        lambda flight: flight[1]["is_departed"] is False)

    t = t0
    while not_departed.count() > 0 and t < t_end:
        candidates = not_departed.filter(
            lambda flight:
            flight[1]["scheduled_departure"] + flight[1]["delay"] <= t +
            min(FLIGHT_SEPARATION - 1, 5) and
            flight[1]["ready_time"] is not None and
            flight[1]["ready_time"] <= t
        )

        candidates = candidates.map(lambda candidate: (
            flight_info_bc.value[candidate[0]]["origin"], candidate)).groupByKey()

        resolved = candidates.flatMap(lambda item: resolve_group(item, t))

        state_rdd = state_rdd.union(resolved).reduceByKey(
            merge_states, numPartitions=10)

        if (t - t0) % CHECKPOINT_EVERY == 0:
            state_rdd = state_rdd.persist(StorageLevel.MEMORY_AND_DISK)
            state_rdd.checkpoint()
            state_rdd.count()

        not_departed = state_rdd.filter(
            lambda flight: flight[1]["is_departed"] is False)
        t += 1

    return state_rdd, flight_info_bc


if __name__ == "__main__":
    influence_score_map = {
        "ATL": 52, "ORD": 57, "DFW": 52
    }

    raw_flights_df = spark.read.csv(
        "dataset/flights.csv", header=True, inferSchema=True)

    flights_df = process_raw_flights(raw_flights_df, year=2015, month=8, day=2)
    flights_df = flights_df.persist()

    final_state, flight_info_bc = run_simulation(
        flights_df, influence_score_map, t0=0, t_end=2880, use_influence=False
    )

    remaining_flights = final_state.filter(
        lambda x: x[1]["is_departed"] is False)
    print(remaining_flights.take(10))
