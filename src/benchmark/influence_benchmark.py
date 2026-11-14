"""Module to benchmark the influence estimation algorithms."""
import math
import time
import argparse
import logging
import yaml
import src.jobs.influence_sequential as influence_sequential
import src.jobs.influence_sample_parallel as influence_sample_parallel
import src.jobs.influence_mapreduce as influence_mapreduce
import src.jobs.influence_mapreduce_batched as influence_mapreduce_batched
from src.influence.common import create_graph
from src.utils.spark import create_spark_session
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

SAMPLE_PARALLEL_BLOCKS = True
SAMPLE_ORACLE_BENCHMARK = True
INF_EST_BENCHMARK = True


if __name__ == "__main__":
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
    graph_bc = spark_session.sparkContext.broadcast(graph)

    sc = spark_session.sparkContext

    # warm-up: the first Spark job pays for the start of the executors
    influence_sample_parallel.sample_oracle(sc, graph_bc, ["ATL"], l=1000, t=300)

    if SAMPLE_ORACLE_BENCHMARK:
        logger.info("SampleOracle block size benchmark (inter-sample parallel) for airport ATL")

        for l in [10, 100, 1000, 10000, 100000]:
            tries = 10

            logger.info("L=%d", l)

            for block_size in [10, 100, 500, 1000, 2000, 5000, 10000]:
                if block_size is None:
                    block = max(influence_sample_parallel.MIN_BLOCK,
                                math.ceil(l / (2 * sc.defaultParallelism)))
                    label = f"default ({block})"
                elif block_size > l:
                    continue # a single task, same as block_size == l
                else:
                    block = block_size
                    label = str(block)

                n_tasks = math.ceil(l / block)
                total_time = 0
                min_time = float("inf")

                for _ in range(tries):
                    start_time = time.perf_counter()
                    influence_sample_parallel.sample_oracle(
                        sc, graph_bc, ["ATL"], l=l, t=300, block_size=block_size)
                    elapsed = time.perf_counter() - start_time
                    total_time += elapsed
                    min_time = min(min_time, elapsed)
                logger.info("block_size=%s (%d tasks): %.3f seconds",
                            label, n_tasks, total_time / tries)

            logger.info("--------------------------------------------------")

        logger.info("================================================================")

    if SAMPLE_ORACLE_BENCHMARK:
        logger.info("SampleOracle benchmark for airport ATL")

        for l in [1, 10, 100, 1000, 10000, 100000]:
            total_time = 0
            tries = 10

            logger.info("L=%d", l)

            for _ in range(tries):
                start_time = time.perf_counter()
                influence_sequential.sample_oracle(graph, ["ATL"], l=l, t=300)
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
            logger.info("Sequential: %.2f seconds", total_time / tries)

            total_time = 0
            for _ in range(tries):
                start_time = time.perf_counter()
                influence_sample_parallel.sample_oracle(spark_session.sparkContext, graph_bc, ["ATL"], l=l, t=300)
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
            logger.info("Inter-sample parallel: %.2f seconds", total_time / tries)

            total_time = 0
            for _ in range(tries):
                start_time = time.perf_counter()
                influence_mapreduce.sample_oracle(spark_session, graph_bc, ["ATL"], l, 300, 3)
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
            logger.info("MapReduce: %.2f seconds", total_time / tries)

            logger.info("--------------------------------------------------")

        logger.info("================================================================")

    if INF_EST_BENCHMARK:
        logger.info("InfEst benchmark")

        graph_broadcast = spark_session.sparkContext.broadcast(graph)
        n = len(graph_broadcast.value)
        airports = list(graph_broadcast.value.keys())

        for l in [1, 10, 20, 30]:
            total_time = 0
            tries = 1

            logger.info("L=%d", l)

            for _ in range(tries):
                start_time = time.perf_counter()
                for airport in airports:
                    influence_mapreduce.inf_est(spark_session, graph_bc, [airport], n, 0.1, 3, n_samples=l)
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
            logger.info("MapReduce: %.2f seconds", total_time / tries)

            total_time = 0
            for _ in range(tries):
                start_time = time.perf_counter()
                influence_mapreduce_batched.inf_est_batched(spark_session, graph_bc, airports, n, 0.1, 3, n_samples=l)
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
            logger.info("MapReduce batched: %.2f seconds", total_time / tries)

            logger.info("--------------------------------------------------")
