"""Module with common methods for influence jobs"""

from pyspark.sql import functions as F

DELAY_THRESHOLD = 15
NEW, OLD = 1, 0


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
    """Merge labels for the same node: OLD prevails."""
    if label1 == OLD or label2 == OLD:
        return OLD
    return NEW
