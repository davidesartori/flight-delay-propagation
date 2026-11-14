"""Spark session utility."""
from pyspark.sql import SparkSession

TARGET_PARTITION_BYTES = 800 * 1024


def create_spark_session(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )
