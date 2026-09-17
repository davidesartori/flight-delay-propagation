"""This module provides functionality to start a Spark Structured Streaming job that reads
random delay datafrom CSV files and stores it in a buffer for further processing."""

import threading
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

random_delay_schema = StructType([
    StructField("airport", StringType(), True),
    StructField("extra_delay", IntegerType(), True),
])


def start_delay_stream(spark_session, input_dir="delay_streaming"):
    """Starts a Spark Structured Streaming job that reads random delay data from CSV files"""
    buffer = {}
    lock = threading.Lock()

    def process_batch(batch_df, _):
        rows = batch_df.collect()
        with lock:
            for row in rows:
                buffer.setdefault(row["airport"], row["extra_delay"])

    stream_df = (
        spark_session.readStream
        .format("csv")
        .schema(random_delay_schema)
        .option("header", "true")
        .option("maxFilesPerTrigger", 1)
        .load(input_dir)
    )

    query = stream_df.writeStream.foreachBatch(process_batch).start()

    return query, buffer, lock


def consume_pending_delays(buffer, lock):
    """Consumes the pending delays from the buffer and returns them as a dictionary."""
    with lock:
        pending = dict(buffer)
        buffer.clear()
    return pending
