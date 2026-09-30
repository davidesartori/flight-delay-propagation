"""Optimized training script for calculating influence scores of airports.
This version processes all airports together at each level of the InfEst algorithm.
"""
import os
import logging
import argparse
import random
import pickle
import yaml
from pyspark import StorageLevel
from src.utils.spark import create_spark_session, compute_num_slices
from src.utils.logging import setup_logging
from src.influence.common import create_graph, estimate_rdd_size_for_epoch, compute_avg_out_degree, merge_labels

logger = logging.getLogger(__name__)

EPSILON = 0.1
MAX_EPOCHS = 3
DELAY_THRESHOLD = 15
N_SAMPLES = 20


def infect_batched(row, graph_bc, t):
    """Infection step for a single (airport, sample_id, node) tuple."""
    (airport, sample_id, u), _ = row
    graph = graph_bc.value
    out_neighbors = graph.get(u, [])[:t]

    results = []
    for v, infection_probability in out_neighbors:
        if random.random() < infection_probability:
            results.append(((airport, sample_id, v), "new"))

    results.append(((airport, sample_id, u), "old"))
    return results


def sample_oracle_batched(spark_session, graph_bc, active_airports, l, t, max_epochs,
                          avg_out_degree, n_total_nodes, node_size):
    """Runs the sampling oracle for all airports in `active_airports` at once"""
    sc = spark_session.sparkContext
    initial_num_slices = compute_num_slices(spark_session, len(active_airports) * l, node_size)

    infected_nodes_rdd = (
        sc.parallelize(active_airports, numSlices=initial_num_slices)
        .flatMap(lambda airport: [((airport, sample_id, airport), "new") for sample_id in range(l)])
    )

    n_incomplete = len(active_airports) * l
    counts_over_threshold = {airport: 0 for airport in active_airports}

    epoch = 0
    prev_persisted = []

    while n_incomplete > 0 and epoch < max_epochs:
        estimated_size = estimate_rdd_size_for_epoch(
            n_incomplete, avg_out_degree, n_total_nodes, epoch + 1
        )
        epoch_num_partitions = compute_num_slices(spark_session, estimated_size, node_size)

        r_d = (
            infected_nodes_rdd
            .flatMap(lambda row: infect_batched(row, graph_bc, t) if row[1] == "new" else [row])
            .reduceByKey(merge_labels, numPartitions=epoch_num_partitions)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        status = (
            r_d # ((sample_id, node), "new")
            .map(lambda row: ((row[0][0], row[0][1]), (1, row[1] == "old"))) # (sample_id, node) -> (1, is_old)
            .reduceByKey(lambda a, b: (a[0] + b[0], a[1] and b[1]))
            .mapValues(lambda v: (v[0], v[0] >= t or v[1]))
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        summary = (
            status
            .map(lambda kv: (
                kv[0][0], # sample_id
                (0 if kv[1][1] else 1, # incomplete sample
                 1 if (kv[1][1] and kv[1][0] >= t) else 0) # over threshold
            ))
            .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
            .collect()
        )

        n_incomplete = 0
        for airport, (inc, over) in summary:
            n_incomplete += inc
            counts_over_threshold[airport] += over

        completed_keys = status.filter(lambda kv: kv[1][1]).map(lambda kv: (kv[0], None))

        infected_nodes_rdd = (
            r_d
            .map(lambda row: ((row[0][0], row[0][1]), row))
            .subtractByKey(completed_keys, numPartitions=epoch_num_partitions)
            .map(lambda kv: kv[1])
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        for rdd in prev_persisted:
            rdd.unpersist()
        prev_persisted = [r_d, status, infected_nodes_rdd]

        epoch += 1

    for rdd in prev_persisted:
        rdd.unpersist()

    return {airport: counts_over_threshold[airport] / l for airport in active_airports}


def verify_guess_batched(spark_session, graph_bc, active_airports, n, tau, epsilon, max_epochs, avg_out_degree, node_size):
    """Runs VerifyGuess for all airports in `active_airports` at once."""
    totals = {airport: 0.0 for airport in active_airports}
    still_undecided = set(active_airports)
    accepted = set()

    t = tau
    while t <= n and still_undecided:
        l = N_SAMPLES

        pi_t_l_by_airport = sample_oracle_batched(
            spark_session, graph_bc, list(still_undecided), l, int(round(t)), max_epochs, avg_out_degree, n, node_size
        )

        for airport in list(still_undecided):
            totals[airport] += (epsilon / (1 + epsilon)) * t * pi_t_l_by_airport[airport]

            if totals[airport] >= (1 - 2 * epsilon) * tau:
                accepted.add(airport)
                still_undecided.discard(airport)

        t = t * (1 + epsilon)

    return {airport: (1 if airport in accepted else 0) for airport in active_airports}


def inf_est_batched(spark_session, graph_bc, airports, n, epsilon, max_epochs, avg_out_degree, node_size):
    """Runs InfEst for all airports at once: at every level"""
    results = {}
    active_airports = set(airports)
    total_airports = len(airports)
    tau = n

    while active_airports and tau >= 1:
        verified = verify_guess_batched(
            spark_session, graph_bc, list(active_airports), n, tau, epsilon, max_epochs, avg_out_degree, node_size
        )

        for airport, is_accepted in verified.items():
            if is_accepted == 1:
                results[airport] = tau
                active_airports.discard(airport)
                logger.info(
                    "Airport: %s, Influence Score: %s (%d/%d)",
                    airport, tau, len(results), total_airports
                )

        tau = tau / (1 + epsilon)

    for airport in active_airports:
        results[airport] = 1
        logger.info(
            "Airport: %s, Influence Score: %s (%d/%d) [fallback]",
            airport, 1, len(results), total_airports
        )

    return results


def main():
    """Main function to run the training script."""
    setup_logging()

    spark_session = create_spark_session(app_name="Training")
    spark_session.sparkContext.setCheckpointDir("checkpoint_dir")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    file = config["input"]["flights"]
    data = spark_session.read.format("csv")\
        .option("header", "true")\
        .option("inferSchema", "true")\
        .load(file)

    graph = create_graph(data)
    graph_broadcast = spark_session.sparkContext.broadcast(graph)
    avg_out_degree = compute_avg_out_degree(graph)

    n = len(graph_broadcast.value)
    airports = list(graph_broadcast.value.keys())

    logger.info(
        "Starting batched influence estimation for %d airports",
        len(airports)
    )

    node_size = len(pickle.dumps((("ATL", 0, "ORD"), "new")))

    influence_scores = inf_est_batched(
        spark_session, graph_broadcast, airports, n, EPSILON,
        max_epochs=MAX_EPOCHS, avg_out_degree=avg_out_degree, node_size=node_size
    )

    logger.info("Finished influence estimation for %d airports", len(influence_scores))

    output_file = config["output"]["training"]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("airport,score\n")
        for airport, score in influence_scores.items():
            f.write(f"{airport},{score}\n")


if __name__ == "__main__":
    main()
