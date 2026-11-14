"""Training script for calculating influence scores of airports.
Versione a epoche (SampleOracle MapReduce): un seed set S alla volta."""
import math
import os
import logging
import argparse
import random
import yaml
from pyspark import StorageLevel
from src.utils.spark import create_spark_session
from src.utils.logging import setup_logging
from src.influence.common import (create_graph,
                                  merge_labels,
                                  NEW, OLD)

logger = logging.getLogger(__name__)

EPSILON = 0.1
MAX_EPOCHS = 3


def infect(row, graph_bc, t):
    """Infection step for a single (sample_id, node) tuple"""
    (sample_id, u), _ = row
    out_neighbors = graph_bc.value.get(u, [])[:t]

    results = []
    for v, infection_probability in out_neighbors:
        if random.random() < infection_probability:
            results.append(((sample_id, v), NEW))

    results.append(((sample_id, u), OLD))
    return results


def sample_oracle(spark_session, graph_bc, starting_nodes, l, t, max_epochs):
    """Fraction of l sampled cascades from the seed set starting_nodes that reach at least t nodes"""
    sc = spark_session.sparkContext

    infected_nodes_rdd = (
        sc.range(l)
        .flatMap(lambda sample_id: [((sample_id, s), NEW) for s in starting_nodes])
    )

    n_incomplete = l
    hits = 0
    epoch = 0
    prev_persisted = []

    while n_incomplete > 0 and epoch < max_epochs:
        r_d = (
            infected_nodes_rdd
            .flatMap(lambda node: infect(node, graph_bc, t) if node[1] == NEW else [node])
            .reduceByKey(merge_labels)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        completed_samples = (
            r_d
            .map(lambda node: (node[0][0], (1, node[1] == OLD)))
            .reduceByKey(lambda a, b: (a[0] + b[0], a[1] and b[1]))
            .filter(lambda kv: kv[1][0] >= t or kv[1][1])
            .mapValues(lambda v: v[0])
            .persist(StorageLevel.MEMORY_AND_DISK)
        ) # (sample_id, n_infected)

        done, ok = (
            completed_samples
            .map(lambda sample: (1, int(sample[1] >= t)))
            .fold((0, 0), lambda a, b: (a[0] + b[0], a[1] + b[1]))
        )
        hits += ok
        n_incomplete -= done

        infected_nodes_rdd = (
            r_d
            .map(lambda node: (node[0][0], node))
            .subtractByKey(completed_samples)
            .map(lambda kv: kv[1])
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        for rdd in prev_persisted:
            rdd.unpersist()
        prev_persisted = [r_d, completed_samples, infected_nodes_rdd]

        epoch += 1

    for rdd in prev_persisted:
        rdd.unpersist()

    return hits / l


def verify_guess(spark_session, graph_bc, starting_nodes, n, tau, epsilon, max_epochs, n_samples=None):
    """Return True if the guess tau of the influence of starting_nodes is accepted"""
    total = 0.0
    t = tau
    while t <= n:
        l = max(10, int(round(8 * t * (math.log(n) ** 3) / (epsilon ** 2 * tau)))) if n_samples is None else n_samples
        pi_t = sample_oracle(
            spark_session, graph_bc, starting_nodes, l, int(round(t)), max_epochs)
        total += (epsilon / (1 + epsilon)) * t * pi_t
        if total >= (1 - 2 * epsilon) * tau:
            return True
        t *= 1 + epsilon
    return False


def inf_est(spark_session, graph_bc, starting_nodes, n, epsilon, max_epochs, n_samples=None):
    """Estimate the influence of the seed set starting_nodes."""
    tau = n
    while tau >= len(starting_nodes):
        if verify_guess(spark_session, graph_bc, starting_nodes, n, tau, epsilon, max_epochs, n_samples):
            return tau
        tau /= 1 + epsilon
    return 1


def main():
    """Main function to run the influence estimation"""
    setup_logging()

    spark_session = create_spark_session(app_name="Training")
    spark_session.sparkContext.setCheckpointDir("checkpoint_dir")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = (spark_session.read.format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .load(config["input"]["flights"]))

    graph = create_graph(data)
    graph_broadcast = spark_session.sparkContext.broadcast(graph)

    n = len(graph_broadcast.value)
    airports = list(graph_broadcast.value.keys())

    logger.info("Starting influence estimation for %d airports", len(airports))

    n_samples = config["influence"].get("n_samples", None) if config["influence"].get("fixed_samples", False) else None

    influence_scores = {}
    for airport in airports:
        try:
            score = inf_est(
                spark_session, graph_broadcast, [airport], n, EPSILON,
                MAX_EPOCHS, n_samples=n_samples
            )
            influence_scores[airport] = score
            logger.info("Airport: %s, Influence Score: %s (%d/%d)",
                        airport, score, len(influence_scores), n)
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
