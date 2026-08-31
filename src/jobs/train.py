"""Training script for calculating influence scores of airports."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import random
import math
from pyspark.sql import functions as F
import yaml
from src.utils.spark import create_spark_session

spark_session, logger, _ = create_spark_session(app_name="Training")

MAX_WORKERS = 10
EPSILON = 0.1
MAX_EPOCHS = 3
DELAY_THRESHOLD = 15


def infect(current_node, graph_bc, t):
    """Simulate the infection process for a given node"""
    (sample_id, u), status = current_node
    graph = graph_bc.value
    out_neighbors = graph.get(u, [])[:t]  # max t neighbors

    results = []

    for v, infecton_probability in out_neighbors:
        if random.random() < infecton_probability:
            results.append(((sample_id, v), "new"))

    results.append(((sample_id, u), "old"))
    return results


def merge_labels(label1, label2):
    """Merge labels for the same node"""
    if label1 == "old" or label2 == "old":
        return "old"
    return "new"


def sample_oracle(graph_bc, s, l, t, max_epochs):
    """Run the sampling oracle to estimate influence"""
    # Initialization
    infected_nodes_list = [
        ((sample_id, node), "new")
        for sample_id in range(l)
        for node in s
    ]

    infected_nodes_rdd = spark_session.sparkContext.parallelize(infected_nodes_list)

    incomplete_samples = set(range(l))
    r_t = {}
    epoch = 1

    while (len(incomplete_samples) > 0 and epoch < max_epochs):
        active_samples_infected_nodes_rdd = infected_nodes_rdd.filter(
            lambda node: node[0][0] in incomplete_samples)
        new_nodes = active_samples_infected_nodes_rdd.filter(
            lambda node: node[1] == 'new')

        t_d = new_nodes.flatMap(lambda row: infect(row, graph_bc, 1000))

        old_nodes = active_samples_infected_nodes_rdd.filter(
            lambda node: node[1] == "old")
        r_d = t_d.union(old_nodes)
        r_d = r_d.reduceByKey(merge_labels)

        infected_nodes_rdd = r_d

        completion_check = (
            r_d
            .map(lambda row: (
                row[0][0],
                (1, row[1] == "old")))
            .reduceByKey(lambda a, b: (
                a[0] + b[0],
                a[1] and b[1]))
            .collect()
        )

        for sample_id, (size, all_old) in completion_check:
            if size >= t or all_old:
                incomplete_samples.discard(sample_id)
                r_t[sample_id] = size

        epoch += 1

    n_over_threshold = sum(1 for size in r_t.values() if size >= t)
    return n_over_threshold / l


def verify_guess(graph_bc, s, n, tau, epsilon, max_epochs):
    """Verify the guess of influence"""
    t = tau
    total = 0.0

    while t <= n:
        l = max(10, math.ceil(8 * t * (math.log(n) ** 3) / (tau ** 2)))
        pi_t_l = sample_oracle(graph_bc, s, l, int(round(t)), max_epochs)
        total += (epsilon / (1 + epsilon)) * t * pi_t_l

        if total >= (1 - 2 * epsilon) * tau:
            return 1

        t = t * (1 + epsilon)

    return 0


def inf_est(graph_bc, s, n, epsilon, max_epochs):
    """Estimate the influence of the given starting nodes s"""
    tau = n

    while tau >= len(s):
        if verify_guess(graph_bc, s, n, tau, epsilon, max_epochs) == 1:
            return tau
        tau = tau / (1 + epsilon)

    return 1


if __name__ == "__main__":
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

    route_delay_probability = (
        data
        .withColumn("is_delayed", F.when(F.col("DEPARTURE_DELAY") > DELAY_THRESHOLD, 1).otherwise(0))
        .groupBy("ORIGIN_AIRPORT", "DESTINATION_AIRPORT")
        .agg(
            F.count("*").alias("n_flights"),
            F.sum("is_delayed").alias("n_delayed")
        )
        .withColumn("delay_ratio", F.col("n_delayed") / F.col("n_flights"))
    )

    edges = route_delay_probability.select(
        "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "delay_ratio"
    ).collect()

    graph = defaultdict(list)

    for row in edges:
        origin = row["ORIGIN_AIRPORT"]
        dest = row["DESTINATION_AIRPORT"]
        prob = row["delay_ratio"]
        graph[origin].append((dest, prob))

    graph = {
        origin: sorted(neighbors, key=lambda x: x[1], reverse=True)
        for origin, neighbors in graph.items()
    }

    graph_broadcast = spark_session.sparkContext.broadcast(graph)


    spark_session.sparkContext.setLocalProperty("spark.scheduler.mode", "FAIR")
    n = len(graph_broadcast.value)

    airports = list(graph_broadcast.value.keys())
    influence_scores = {}


    def run_single(airport):
        """Run the influence estimation for a single airport."""
        s = [airport]
        score = inf_est(graph_broadcast, s, n, EPSILON, max_epochs=MAX_EPOCHS)
        return airport, score


    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single, airport): airport for airport in airports}

        for future in as_completed(futures):
            airport = futures[future]
            try:
                airport_result, score = future.result()
                influence_scores[airport_result] = score
                print(f"{airport_result}: {score:.3f}  ({len(influence_scores)}/{n})")
            except Exception as e:
                print(f"ERRORE su {airport}: {e}")

    output_file = config["output"]["training"]
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("airport,influence_score\n")
        for airport, score in influence_scores.items():
            f.write(f"{airport},{score}\n")
