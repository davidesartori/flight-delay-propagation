"""Training script for influence estimation: InfEst with
sample-parallel SampleOracle (local BFS per task)."""
import os
import logging
import argparse
import random
import math
import yaml
from src.utils.spark import create_spark_session
from src.influence.common import create_graph
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

EPSILON = 0.1
MAX_EPOCHS = 3
MIN_BLOCK = 500


def simulate(graph, starting_nodes, t):
    """One sample: True if the cascade from the starting_nodes
    reaches at least t nodes within MAX_EPOCHS epochs"""
    infected_nodes = set(starting_nodes)
    infecting_nodes = list(starting_nodes)
    if len(infected_nodes) >= t:
        return True

    for _ in range(MAX_EPOCHS):
        new_nodes = []
        for u in infecting_nodes:
            for v, probability in graph.get(u, [])[:t]:  # at most t neighbors
                if v not in infected_nodes and random.random() < probability:
                    infected_nodes.add(v)
                    new_nodes.append(v)
        if len(infected_nodes) >= t:
            return True
        if not new_nodes:
            return False
        infecting_nodes = new_nodes
    return False


def run_block(size, graph_bc, starting_nodes, t):
    """Run a block of samples in parallel, returning the number of hits"""
    graph = graph_bc.value
    return sum(1 for _ in range(size) if simulate(graph, starting_nodes, t))


def sample_oracle(sc, graph_bc, starting_nodes, l, t, block_size=None):
    """Returns the fraction of L sampled cascades from
    starting_nodes that reach at least t nodes."""
    n_blocks_min = 2 * sc.defaultParallelism # 2 tasks per core
    block_max = math.ceil(l / n_blocks_min)
    block = block_size if block_size is not None else max(MIN_BLOCK, block_max)

    sizes = [block] * (l // block)
    if l % block:
        sizes.append(l % block)
    hits = (sc.parallelize(sizes, len(sizes))
              .map(lambda size: run_block(size, graph_bc, starting_nodes, t))
              .sum())
    return hits / l


def verify_guess(sc, graph_bc, starting_nodes, tau, epsilon, n_samples=None):
    """Return True if the guess tau of the influence of starting_nodes is accepted"""
    n = len(graph_bc.value)
    t = tau
    total = 0.0
    while t <= n:
        l = max(10, math.ceil(8 * t * (math.log(n) ** 3) / (epsilon ** 2 * tau))) if n_samples is None else n_samples
        pi_t = sample_oracle(sc, graph_bc, starting_nodes, l, int(round(t)))
        total += (epsilon / (1 + epsilon)) * t * pi_t
        if total >= (1 - 2 * epsilon) * tau:
            return True
        t *= 1 + epsilon
    return False


def inf_est(sc, graph_bc, starting_nodes, epsilon, n_samples=None):
    """Estimate the influence of starting_nodes"""
    n = len(graph_bc.value)
    tau = n
    while tau >= len(starting_nodes):
        if verify_guess(sc, graph_bc, starting_nodes, tau, epsilon, n_samples):
            return tau
        tau /= 1 + epsilon
    return 1


def main():
    """Main function to run the influence estimation"""
    setup_logging()
    spark_session = create_spark_session(app_name="Training")
    sc = spark_session.sparkContext

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = (spark_session.read.format("csv")
            .option("header", "true").option("inferSchema", "true")
            .load(config["input"]["flights"]))

    graph = create_graph(data)
    graph_bc = sc.broadcast(graph)
    n = len(graph)
    airports = list(graph.keys())

    logger.info("Starting influence estimation for %d airports", len(airports))

    n_samples = config["influence"].get("n_samples", None) if config["influence"].get("fixed_samples", False) else None

    influence_scores = {}
    for airport in airports:
        try:
            score = inf_est(sc, graph_bc, [airport], EPSILON, n_samples)
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
