"""Training script for influence estimation using sequential InfEst."""
import os
import math
import logging
import random
import argparse
import yaml
from src.utils.spark import create_spark_session
from src.influence.common import create_graph
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

EPSILON = 0.1
MAX_EPOCHS = 3


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


def sample_oracle(graph, starting_nodes, l, t):
    """Returns the fraction of L sampled cascades from
    starting_nodes that reach at least t nodes"""
    hits = 0
    for _ in range(l):
        if simulate(graph, starting_nodes, t):
            hits += 1
    return hits / l


def verify_guess(graph, starting_nodes, tau, epsilon):
    """Return True if the guess tau of the influence of starting_nodes is accepted"""
    n = len(graph)
    t = tau
    total = 0.0
    while t <= n:
        l = max(10, int(round(8 * t * (math.log(n) ** 3) / (epsilon ** 2 * tau))))
        pi_t = sample_oracle(graph, starting_nodes, l, int(round(t)))
        total += (epsilon / (1 + epsilon)) * t * pi_t
        if total >= (1 - 2 * epsilon) * tau:
            return True
        t *= 1 + epsilon
    return False


def inf_est(graph, starting_nodes, epsilon):
    """Estimate the influence of starting_nodes"""
    n = len(graph)
    tau = n
    while tau >= len(starting_nodes):
        if verify_guess(graph, starting_nodes, tau, epsilon):
            return tau
        tau /= 1 + epsilon
    return 1


def main():
    """Main function to run the influence estimation"""
    setup_logging()
    spark_session = create_spark_session(app_name="Training")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = (spark_session.read.format("csv")
            .option("header", "true").option("inferSchema", "true")
            .load(config["input"]["flights"]))

    graph = create_graph(data)
    n = len(graph)
    airports = list(graph.keys())

    logger.info("Starting influence estimation for %d airports", len(airports))

    influence_scores = {}
    for airport in airports:
        try:
            score = inf_est(graph, [airport], EPSILON)
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
