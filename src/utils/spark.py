"""Spark session utility."""
import math
from pyspark.sql import SparkSession

TARGET_PARTITION_BYTES = 800 * 1024


def create_spark_session(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )


def compute_num_slices(spark_session, n_elements, size, cap_multiplier=10):
    """Return a number of partitions proportional to the actual
    (or estimated) size of the data"""
    default_parallelism = spark_session.sparkContext.defaultParallelism
    rows_per_partition = max(1, TARGET_PARTITION_BYTES // size)
    num_slices = math.ceil(n_elements / rows_per_partition)
    num_slices = max(num_slices, default_parallelism)
    num_slices = min(num_slices, default_parallelism * cap_multiplier)
    return num_slices
