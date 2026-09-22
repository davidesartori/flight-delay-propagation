"""Module for computing simulation statistics."""


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


def compute_simulation_stats(final_state_rdd, flight_info_bc, influence_scores, top_n=20):
    """Computes various statistics from the final state of the simulation."""

    departed = final_state_rdd.filter(lambda item: item[1]["is_departed"])
    not_departed_count = final_state_rdd.filter(
        lambda item: not item[1]["is_departed"]
    ).count()

    def to_record(item):
        flight_id, state = item
        destination = flight_info_bc.value[flight_id]["destination"]
        return (flight_id, state["scheduled_arrival"] - (state["scheduled_departure"] + flight_info_bc.value[flight_id]["scheduled_flying_time"]), destination)

    records = departed.map(to_record)
    records.cache()

    n_departed = records.count()

    if n_departed == 0:
        return _empty_stats(not_departed_count)

    delayed_only = records.filter(lambda r: r[1] >= 1)
    delayed_only.cache()

    delays_gt_60 = delayed_only.filter(lambda r: r[1] >= 60).count()
    delays_gt_120 = delayed_only.filter(lambda r: r[1] >= 120).count()
    delays_gt_200 = delayed_only.filter(lambda r: r[1] >= 200).count()
    delays_gt_250 = delayed_only.filter(lambda r: r[1] >= 250).count()
    delays_gt_300 = delayed_only.filter(lambda r: r[1] >= 300).count()
    delays_gt_400 = delayed_only.filter(lambda r: r[1] >= 400).count()

    total_delay = delayed_only.map(lambda r: r[1]).sum()
    n_delayed = delayed_only.count()

    avg_delay = total_delay / n_departed if n_departed > 0 else 0
    avg_delay_delayed = total_delay / n_delayed if n_delayed > 0 else 0

    all_delays_sorted = sorted(records.map(lambda r: r[1]).collect())
    median_delay = _percentile(all_delays_sorted, 50)

    percentiles = {
        f"p{p}": _percentile(all_delays_sorted, p)
        for p in [50, 75, 90, 95, 99]
    }
    max_delay = all_delays_sorted[-1] if all_delays_sorted else 0

    ranked_airports = sorted(influence_scores.items(), key=lambda kv: kv[1], reverse=True)
    top_airports = set(a for a, _ in ranked_airports[:top_n])
    top_airports_bc = records.context.broadcast(top_airports)

    top_records = records.filter(lambda r: r[2] in top_airports_bc.value)
    top_records.cache()
    n_top = top_records.count()

    if n_top > 0:
        top_delays_sorted = sorted(top_records.map(lambda r: r[1]).collect())
        top_avg_delay = sum(top_delays_sorted) / n_top
        top_median_delay = _percentile(top_delays_sorted, 50)
        top_total_delay = sum(d for d in top_delays_sorted if d >= 1)
    else:
        top_avg_delay = top_median_delay = top_total_delay = 0

    influence_bc_local = records.context.broadcast(influence_scores)

    def weighted_delay(r):
        _, delay, destination = r
        if delay < 1:
            return 0.0
        influence = influence_bc_local.value.get(destination, 0)
        return delay * influence

    total_weighted_delay = records.map(weighted_delay).sum()

    return {
        "total_delay": total_delay,
        "n_delayed_flights": n_delayed,
        "n_delays_gt_60": delays_gt_60,
        "n_delays_gt_120": delays_gt_120,
        "n_delays_gt_200": delays_gt_200,
        "n_delays_gt_250": delays_gt_250,
        "n_delays_gt_300": delays_gt_300,
        "n_delays_gt_400": delays_gt_400,
        "n_departed_flights": n_departed,
        "n_not_departed": not_departed_count,
        "avg_delay": avg_delay,
        "avg_delay_delayed": avg_delay_delayed,
        "median_delay": median_delay,
        "percentiles": percentiles,
        "max_delay": max_delay,
        "n_top_airport_flights": n_top,
        "top_avg_delay": top_avg_delay,
        "top_median_delay": top_median_delay,
        "top_total_delay": top_total_delay,
        "total_weighted_delay": total_weighted_delay,
    }


def _percentile(sorted_values, p):
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, int(round((p / 100) * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _empty_stats(not_departed_count: int):
    empty_percentiles = {f"p{p}": 0 for p in [50, 75, 90, 95, 99]}
    return {
        "total_delay": 0, "n_delayed_flights": 0, "n_departed_flights": 0,
        "n_not_departed": not_departed_count, "avg_delay": 0,
        "avg_delay_delayed": 0, "median_delay": 0,
        "percentiles": empty_percentiles, "max_delay": 0,
        "n_top_airport_flights": 0, "top_avg_delay": 0,
        "top_median_delay": 0, "top_total_delay": 0, "total_weighted_delay": 0,
    }
