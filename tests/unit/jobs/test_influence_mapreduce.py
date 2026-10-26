"""Testing module for influence job"""
import random
import pytest
from src.jobs import influence_mapreduce
from src.influence.common import NEW, OLD


@pytest.fixture
def flights_df(spark):
    """Small flight dataset with known delay ratios per route."""
    data = [
        ("A", "B", 20),
        ("A", "B", 5),
        ("A", "B", 16),
        ("A", "B", 0),
        ("A", "C", 1),
        ("A", "C", 2),
        ("B", "C", 30),
        ("B", "C", 16),
        ("B", "C", 90),
    ]
    return spark.createDataFrame(
        data, ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DEPARTURE_DELAY"]
    )


@pytest.fixture
def toy_graph():
    """Small, Spark-independent toy graph."""
    return {
        "A": [("B", 0.9), ("C", 0.1)],
        "B": [("C", 0.5)],
        "C": [],
    }


@pytest.fixture
def broadcast_factory(spark):
    """Creates a real Spark broadcast for a given graph."""
    def _make(graph):
        return spark.sparkContext.broadcast(graph)
    return _make


class TestInfect:
    """Tests for the infect function."""

    def test_always_infects_when_random_below_probability(
        self, monkeypatch, toy_graph, broadcast_factory
    ):
        """All neighbors should get infected when random() is always below their probability."""
        monkeypatch.setattr(random, "random", lambda: 0.0)
        bc = broadcast_factory(toy_graph)
        current = (("sample0", "A"), NEW)
        results = influence_mapreduce.infect(current, bc, t=10)

        new_nodes = {node for node, label in results if label == NEW}
        old_nodes = {node for node, label in results if label == OLD}

        assert new_nodes == {("sample0", "B"), ("sample0", "C")}
        assert old_nodes == {("sample0", "A")}

    def test_never_infects_when_random_above_probability(
        self, monkeypatch, toy_graph, broadcast_factory
    ):
        """No neighbors should get infected when random() is always above their probability."""
        monkeypatch.setattr(random, "random", lambda: 1)
        bc = broadcast_factory(toy_graph)
        current = (("sample0", "A"), NEW)
        results = influence_mapreduce.infect(current, bc, t=10)

        assert results == [(("sample0", "A"), OLD)]

    def test_respects_max_neighbors_t(self, monkeypatch, broadcast_factory):
        """Only the first t neighbors should be considered for infection."""
        monkeypatch.setattr(random, "random", lambda: 0.0)
        graph = {"A": [("B", 0.9), ("C", 0.8), ("D", 0.7)]}
        bc = broadcast_factory(graph)
        current = (("sample0", "A"), NEW)

        results = influence_mapreduce.infect(current, bc, t=2)
        new_nodes = {node for node, label in results if label == NEW}
        assert new_nodes == {("sample0", "B"), ("sample0", "C")}

    def test_node_with_no_outgoing_edges_only_returns_old_self(
        self, broadcast_factory
    ):
        """A node with no outgoing edges should only return itself, marked as old."""
        bc = broadcast_factory({})
        current = (("sample0", "Z"), NEW)
        results = influence_mapreduce.infect(current, bc, t=5)
        assert results == [(("sample0", "Z"), OLD)]


class TestSampleOracle:
    """Tests for the sample_oracle function."""

    def test_returns_fraction_between_0_and_1(
        self, spark, toy_graph, broadcast_factory
    ):
        """sample_oracle should return a fraction between 0 and 1."""
        bc = broadcast_factory(toy_graph)
        result = influence_mapreduce.sample_oracle(spark, bc, ["A"], l=5, t=2, max_epochs=3)
        assert 0.0 <= result <= 1.0

    def test_isolated_node_never_exceeds_threshold_gt_1(
        self, spark, broadcast_factory
    ):
        """An isolated seed node should never reach a threshold greater than 1."""
        bc = broadcast_factory({"Z": []})
        result = influence_mapreduce.sample_oracle(spark, bc, ["Z"], l=3, t=2, max_epochs=3)
        assert result == pytest.approx(0.0)


class TestVerifyGuessAndInfEst:
    """Tests for the verify_guess and inf_est functions."""

    def test_verify_guess_returns_0_or_1(
        self, spark, toy_graph, broadcast_factory
    ):
        """verify_guess should always return either 0 or 1."""
        bc = broadcast_factory(toy_graph)
        result = influence_mapreduce.verify_guess(
            spark, bc, ["A"], n=3, tau=1, epsilon=influence_mapreduce.EPSILON, max_epochs=3
        )
        assert result in (0, 1)

    def test_inf_est_at_least_size_of_seed_set(
        self, spark, toy_graph, broadcast_factory
    ):
        """inf_est should never return a score smaller than the seed set size."""
        bc = broadcast_factory(toy_graph)
        score = influence_mapreduce.inf_est(spark, bc, ["A"], n=3, epsilon=influence_mapreduce.EPSILON, max_epochs=3)
        assert score >= 1
