"""Run the simulation on several days, with and without influence, and
summarize how every statistic changes.

Usage:
    python -m src.jobs.simulation_benchmark --config config/config.yaml \
        --year 2015 --month 8
"""
import argparse
import calendar
import json
import logging
import math
import os
import re
import statistics
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml
from pyspark.sql import functions as F

from src.jobs.simulation import process_raw_flights, T_START, T_MAX, run_simulation
from src.utils.io import load__influence_scores
from src.utils.logging import setup_logging
from src.utils.spark import create_spark_session

logger = logging.getLogger(__name__)

HIGHER_IS_BETTER = {"n_departed_flights"}

# top-N airports (by delay) to aggregate from stats["top_delays"]
TOP_NS = (1, 5, 10, 20)
RANKING_SUM_FIELDS = (
    "n_delayed", "total_delay",
    "n_delays_gt_60", "n_delays_gt_120", "n_delays_gt_200",
    "n_delays_gt_250", "n_delays_gt_300", "n_delays_gt_400",
)


def top_n_stats(ranking, ns=TOP_NS):
    """Aggregate the per-airport ranking (already ordered by delay) over the
    first N airports, for each N in ns. Keys look like 'worst5.total_delay'."""
    flat = {}
    for n in ns:
        top = ranking[:n]
        if not top:
            continue
        prefix = f"worst{n}"
        n_flights = sum(r.get("n_flights", 0) for r in top)
        total_delay = sum(r.get("total_delay", 0) for r in top)
        for field in RANKING_SUM_FIELDS:
            flat[f"{prefix}.{field}"] = float(sum(r.get(field, 0) for r in top))
        flat[f"{prefix}.avg_delay"] = total_delay / n_flights if n_flights else 0.0
        flat[f"{prefix}.max_delay"] = float(max(r.get("max_delay", 0) for r in top))
    return flat


def flatten_stats(stats):
    """Keep only numeric values; nested dicts (percentiles) become 'a.b' keys.
    The per-airport ranking is summarized for the top 1/5/10/20 airports."""
    flat = {}
    for key, value in stats.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            flat[key] = float(value)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)) and not isinstance(sub_value, bool):
                    flat[f"{key}.{sub_key}"] = float(sub_value)
    flat.update(top_n_stats(stats.get("top_delays") or []))
    return flat


def append_result(path, day, use_influence, stats):
    """Append one finished run to the results file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"day": day, "influence": use_influence, "stats": stats}) + "\n")


def run_day(spark, flights_df, influence_scores, year, month, day,
            use_influence, influence_rate):
    """Run one simulation and return its flattened statistics."""
    _, stats, _ = run_simulation(
        spark, flights_df, influence_scores,
        date=datetime(year, month, day, tzinfo=ZoneInfo("America/Chicago")),
        t0=T_START, t_end=T_MAX,
        use_influence=use_influence, influence_rate=influence_rate)
    flat = flatten_stats(stats)
    spark.catalog.clearCache() # compute_simulation_stats caches DataFrames
    return flat


def sort_key(key):
    """Global statistics first, then worst1, worst5, worst10, worst20 in numeric order."""
    m = re.match(r"worst(\d+)\.", key)
    return (1, int(m.group(1)), key) if m else (0, 0, key)


def print_summary(results, days):
    """Print, for every statistic, the average change with influence vs without.
    'Mean %' is the change of the mean: mean diff / mean without influence."""
    paired = [d for d in days if (d, False) in results and (d, True) in results]
    if not paired:
        print("No day has both runs (with and without influence): nothing to compare.")
        return

    keys = sorted(set().union(*[results[(d, True)].keys() for d in paired]),
                  key=sort_key)

    header = (f"{'Statistic':<28}{'Mean w/o':>14}{'Mean with':>14}{'Mean diff':>13}"
              f"{'Mean %':>9}{'Better':>8}{'Worse':>7}{'Same':>6}{'t':>8}")
    print(f"\nInfluence ON vs OFF over {len(paired)} days "
          f"({paired[0]}..{paired[-1]})")
    print("diff = with - without; 'Better' = days in which influence did better")
    print("Mean % = mean diff / mean w/o * 100")
    print(header)
    print("-" * len(header))

    for key in keys:
        base_vals, inf_vals, diffs = [], [], []
        better = worse = same = 0
        for d in paired:
            base = results[(d, False)].get(key)
            with_inf = results[(d, True)].get(key)
            if base is None or with_inf is None:
                continue
            diff = with_inf - base
            base_vals.append(base)
            inf_vals.append(with_inf)
            diffs.append(diff)
            if diff == 0:
                same += 1
            elif (diff > 0) == (key in HIGHER_IS_BETTER):
                better += 1
            else:
                worse += 1
        if not diffs:
            continue

        mean_diff = statistics.fmean(diffs)
        base_mean = statistics.fmean(base_vals)
        mean_rel = f"{mean_diff / base_mean * 100:+.1f}%" if base_mean else "n/a"

        t_stat = "n/a"
        if len(diffs) > 1 and statistics.stdev(diffs) > 0:
            t_stat = f"{mean_diff / (statistics.stdev(diffs) / math.sqrt(len(diffs))):+.2f}"

        print(f"{key:<28}{base_mean:>14.2f}"
              f"{statistics.fmean(inf_vals):>14.2f}{mean_diff:>+13.2f}"
              f"{mean_rel:>9}{better:>8}{worse:>7}{same:>6}{t_stat:>8}")

    print("\nt = mean diff / standard error: |t| above ~2 suggests the change is "
          "not just day-to-day noise (rough guide, small samples).")


def main():
    """Run all simulations and print the summary."""
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--month", type=int, default=8)
    parser.add_argument("--first-day", type=int, default=1)
    parser.add_argument("--last-day", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    last_day = args.last_day or calendar.monthrange(args.year, args.month)[1]
    days = list(range(args.first_day, last_day + 1))

    spark = create_spark_session(app_name="CompareInfluence")

    influence_scores = load__influence_scores(config["input"]["influence"])
    influence_rate = config["simulation"].get("influence_rate", 2)

    raw_df = (spark.read.csv(config["input"]["flights"], header=True, inferSchema=True)
              .filter((F.col("YEAR") == args.year) & (F.col("MONTH") == args.month))
              .persist())

    results = {}

    for day in days:
        todo = [mode for mode in (False, True) if (day, mode) not in results]
        if not todo:
            logger.info("Day %d already done, skipping", day)
            continue

        flights_df = process_raw_flights(
            spark, raw_df, year=args.year, month=args.month, day=day).persist()

        for use_influence in todo:
            logger.info("Day %d/%d, influence=%s", day, days[-1], use_influence)
            try:
                stats = run_day(spark, flights_df, influence_scores, args.year,
                                args.month, day, use_influence, influence_rate)
            except Exception:
                logger.exception("Run failed for day %d, influence=%s", day, use_influence)
                continue
            results[(day, use_influence)] = stats

        flights_df.unpersist()

    print_summary(results, days)


if __name__ == "__main__":
    main()
