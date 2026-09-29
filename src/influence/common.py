"""Module with common methods for influence jobs"""

from pyspark.sql import functions as F

DELAY_THRESHOLD = 15


def estimate_rdd_size_for_epoch(n_active_samples, avg_out_degree, n_total_nodes, epoch):
    """Upper-bound estimate of the number of elements in the infected RDD
    at a given epoch of sample_oracle_batched."""
    estimated_per_sample = min(avg_out_degree ** epoch, n_total_nodes)
    return n_active_samples * estimated_per_sample


def compute_avg_out_degree(graph):
    """upper-bound estimate of the branching factor,
    used to size Spark partitions during the infection loop."""
    if not graph:
        return 0.0
    return sum(len(neighbors) for neighbors in graph.values()) / len(graph)


def create_graph(data):
    """Create a directed graph from the flight data with delay probabilities."""
    route_delay_probability = (
        data
        .withColumn("is_delayed", F.when(F.col("DEPARTURE_DELAY") >
                                         DELAY_THRESHOLD, 1).otherwise(0))
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

    graph = {}
    for row in edges:
        origin = row["ORIGIN_AIRPORT"]
        dest = row["DESTINATION_AIRPORT"]
        prob = row["delay_ratio"]
        graph.setdefault(origin, []).append((dest, prob))

    graph = {
        origin: sorted(neighbors, key=lambda x: x[1], reverse=True)
        for origin, neighbors in graph.items()
    }

    return graph


def merge_labels(label1, label2):
    """Merge labels for the same node"""
    if label1 == "old" or label2 == "old":
        return "old"
    return "new"
