"""Training script for calculating influence scores of airports."""
import os
import logging
import argparse
import random
import math
import pickle
from pyspark import StorageLevel
import yaml
from src.utils.spark import create_spark_session, compute_num_slices
from src.influence.common import create_graph, estimate_rdd_size_for_epoch, compute_avg_out_degree, merge_labels
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

EPSILON = 0.1
MAX_EPOCHS = 3


def infect(current_node, graph_bc, t):
    """Simulate the infection process for a given node"""
    (sample_id, u), _ = current_node
    graph = graph_bc.value
    out_neighbors = graph.get(u, [])[:t]  # max t neighbors

    results = []

    for v, infecton_probability in out_neighbors:
        if random.random() < infecton_probability:
            results.append(((sample_id, v), "new"))

    results.append(((sample_id, u), "old"))
    return results


def sample_oracle(spark_session, graph_bc, s, l, t, max_epochs, avg_out_degree, n_total_nodes, node_size):
    """Run the sampling oracle to estimate influence"""
    initial_num_slices = compute_num_slices(spark_session, l, node_size)
    infected_nodes_rdd = (spark_session.sparkContext.range(l, numSlices=initial_num_slices)
                        .flatMap(lambda sample_id: [((sample_id, node), "new") for node in s ]))

    incomplete_samples = (spark_session.sparkContext.range(l)
                        .map(lambda sample_id: (sample_id, None)))

    n_incomplete = l
    n_over_threshold = 0
    epoch = 0

    while n_incomplete > 0 and epoch < max_epochs:
        active_samples_infected_nodes_rdd = (infected_nodes_rdd.map(lambda row: (row[0][0], row))
                                             .join(incomplete_samples)
                                             .map(lambda row: row[1][0]))
        new_nodes = active_samples_infected_nodes_rdd.filter(
            lambda node: node[1] == 'new')

        t_d = new_nodes.flatMap(lambda row: infect(row, graph_bc, t))

        estimated_size = estimate_rdd_size_for_epoch(
            n_incomplete, avg_out_degree, n_total_nodes, epoch + 1
        )
        epoch_num_slices = compute_num_slices(spark_session, estimated_size, node_size)

        old_nodes = active_samples_infected_nodes_rdd.filter(
            lambda node: node[1] == "old")

        r_d = (t_d.union(old_nodes)
               .reduceByKey(merge_labels, numPartitions=epoch_num_slices)
               .persist(StorageLevel.DISK_ONLY))


        sample_stats = (r_d.map(lambda row: (row[0][0], (1, row[1] == "old")))
                        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] and b[1]), numPartitions=epoch_num_slices))

        completed_samples = (sample_stats.filter( lambda row: row[1][0] >= t or row[1][1])
                             .persist(StorageLevel.MEMORY_AND_DISK))

        completed_count, successful_count = (completed_samples.map(lambda row: ( 1, int(row[1][0] >= t)))
                                             .fold((0, 0), lambda a, b: (a[0] + b[0], a[1] + b[1])))

        n_incomplete -= completed_count
        n_over_threshold += successful_count

        next_incomplete_samples = (incomplete_samples.subtractByKey(completed_samples.map(lambda row: (row[0], None)))
                                   .persist(StorageLevel.MEMORY_AND_DISK))

        completed_samples.unpersist()
        incomplete_samples = next_incomplete_samples
        infected_nodes_rdd = r_d

        epoch += 1

    incomplete_samples.unpersist()

    return n_over_threshold / l


def verify_guess(spark_session, graph_bc, s, n, tau, epsilon, max_epochs, avg_out_degree, node_size):
    """Verify the guess of influence"""
    t = tau
    total = 0.0

    while t <= n:
        l = max(10, math.ceil(8 * t * (math.log(n) ** 3) / (epsilon ** 2 * tau)))
        pi_t_l = sample_oracle(spark_session, graph_bc, s, l, int(round(t)), max_epochs, avg_out_degree, n, node_size)
        total += (epsilon / (1 + epsilon)) * t * pi_t_l

        if total >= (1 - 2 * epsilon) * tau:
            return 1

        t = t * (1 + epsilon)

    return 0


def inf_est(spark_session, graph_bc, s, n, epsilon, max_epochs, avg_out_degree, node_size):
    """Estimate the influence of the given starting nodes s"""
    tau = n

    while tau >= len(s):
        if verify_guess(spark_session, graph_bc, s, n, tau, epsilon, max_epochs, avg_out_degree, node_size) == 1:
            return tau
        tau = tau / (1 + epsilon)

    return 1


def main():
    """Main function to run the training script."""
    setup_logging()

    spark_session = create_spark_session(app_name="Training")

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

    logger.info("Starting influence estimation for %d airports", len(airports))

    node_size = len(pickle.dumps(((0, "ORD"), "new")))

    influence_scores = {}
    for airport in airports:
        try:
            s = [airport]
            score = inf_est(
                spark_session, graph_broadcast, s, n, EPSILON,
                max_epochs=MAX_EPOCHS, avg_out_degree=avg_out_degree, node_size=node_size
            )
            influence_scores[airport] = score
            logger.info(
                "Airport: %s, Influence Score: %s (%d/%d)",
                airport, score, len(influence_scores), n
            )
        except Exception:
            logger.exception("Error with %s", airport)

    output_file = config["output"]["training"]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("airport,score\n")
        for airport, score in influence_scores.items():
            f.write(f"{airport},{score}\n")


if __name__ == "__main__":
    main()
