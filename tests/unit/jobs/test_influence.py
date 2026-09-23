"""Testing module for influence job"""
import csv
import random
import textwrap
import pytest
import src.jobs.influence as influence


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
    """Small, Spark-independent toy graph for infect / sample_oracle /
    verify_guess / inf_est tests."""
    return {
        "A": [("B", 0.9), ("C", 0.1)],
        "B": [("C", 0.5)],
        "C": [],
    }


@pytest.fixture
def broadcast_factory(spark):
    """Creates a real Spark broadcast for a given graph (more faithful than a
    plain mock, since the code accesses `.value`)."""
    def _make(graph):
        return spark.sparkContext.broadcast(graph)
    return _make


def test_delay_ratio_computed_correctly(flights_df):
    """Delay ratio per route should equal delayed flights / total flights."""
    graph = influence.create_graph(flights_df)
    neighbors_a = dict(graph["A"])
    assert neighbors_a["B"] == pytest.approx(0.5)
    assert neighbors_a["C"] == pytest.approx(0.0)

def test_neighbors_sorted_descending_by_probability(flights_df):
    """Neighbors of a node should be sorted by delay probability, descending."""
    graph = influence.create_graph(flights_df)
    probs = [p for _, p in graph["A"]]
    assert probs == sorted(probs, reverse=True)

def test_node_with_no_outgoing_edges_absent_as_key(flights_df):
    """A node with no outgoing flights should not appear as a key in the graph."""
    graph = influence.create_graph(flights_df)
    assert "C" not in graph

def test_full_delay_ratio(flights_df):
    """A route where every flight is delayed should have a delay ratio of 1.0."""
    graph = influence.create_graph(flights_df)
    neighbors_b = dict(graph["B"])
    assert neighbors_b["C"] == pytest.approx(1.0)

def test_empty_input_returns_empty_graph(spark):
    """An empty flights dataframe should produce an empty graph."""
    empty_df = spark.createDataFrame(
        [], "ORIGIN_AIRPORT string, DESTINATION_AIRPORT string, DEPARTURE_DELAY int"
    )
    graph = influence.create_graph(empty_df)
    assert graph == {}

def test_delay_threshold_boundary_not_counted(spark):
    """A delay exactly equal to the threshold should not count as delayed."""
    df = spark.createDataFrame(
        [("A", "B", 15), ("A", "B", 15)],
        ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DEPARTURE_DELAY"],
    )
    graph = influence.create_graph(df)
    assert dict(graph["A"])["B"] == pytest.approx(0.0)


class TestMergeLabels:
    """Tests for the merge_labels function."""

    @pytest.mark.parametrize("l1,l2,expected", [
        ("old", "old", "old"),
        ("old", "new", "old"),
        ("new", "old", "old"),
        ("new", "new", "new"),
    ])
    def test_merge_labels_truth_table(self, l1, l2, expected):
        """merge_labels should return 'old' if either label is 'old', else 'new'."""
        assert influence.merge_labels(l1, l2) == expected


class TestInfect:
    """Tests for the infect function."""

    def test_always_infects_when_random_below_probability(
        self, monkeypatch, toy_graph, broadcast_factory
    ):
        """All neighbors should get infected when random() is always below their probability."""
        monkeypatch.setattr(random, "random", lambda: 0.0)
        bc = broadcast_factory(toy_graph)
        current = (("sample0", "A"), "new")
        results = influence.infect(current, bc, t=10)

        new_nodes = {node for node, label in results if label == "new"}
        old_nodes = {node for node, label in results if label == "old"}

        assert new_nodes == {("sample0", "B"), ("sample0", "C")}
        assert old_nodes == {("sample0", "A")}

    def test_never_infects_when_random_above_probability(
        self, monkeypatch, toy_graph, broadcast_factory
    ):
        """No neighbors should get infected when random() is always above their probability."""
        monkeypatch.setattr(random, "random", lambda: 0.999)
        bc = broadcast_factory(toy_graph)
        current = (("sample0", "A"), "new")
        results = influence.infect(current, bc, t=10)

        assert results == [(("sample0", "A"), "old")]

    def test_respects_max_neighbors_t(self, monkeypatch, broadcast_factory):
        """Only the first t neighbors should be considered for infection."""
        monkeypatch.setattr(random, "random", lambda: 0.0)
        graph = {"A": [("B", 0.9), ("C", 0.8), ("D", 0.7)]}
        bc = broadcast_factory(graph)
        current = (("sample0", "A"), "new")

        results = influence.infect(current, bc, t=2)
        new_nodes = {node for node, label in results if label == "new"}
        assert new_nodes == {("sample0", "B"), ("sample0", "C")}

    def test_node_with_no_outgoing_edges_only_returns_old_self(
        self, broadcast_factory
    ):
        """A node with no outgoing edges should only return itself, marked as old."""
        bc = broadcast_factory({})
        current = (("sample0", "Z"), "new")
        results = influence.infect(current, bc, t=5)
        assert results == [(("sample0", "Z"), "old")]


class TestSampleOracle:
    """Tests for the sample_oracle function."""

    def test_returns_fraction_between_0_and_1(
        self, spark, toy_graph, broadcast_factory
    ):
        """sample_oracle should return a fraction between 0 and 1."""
        bc = broadcast_factory(toy_graph)
        result = influence.sample_oracle(spark, bc, s=["A"], l=5, t=2, max_epochs=3)
        assert 0.0 <= result <= 1.0

    def test_isolated_node_never_exceeds_threshold_gt_1(
        self, spark, broadcast_factory
    ):
        """An isolated seed node should never reach a threshold greater than 1."""
        bc = broadcast_factory({"Z": []})
        result = influence.sample_oracle(spark, bc, s=["Z"], l=3, t=2, max_epochs=3)
        assert result == pytest.approx(0.0)


class TestVerifyGuessAndInfEst:
    """Tests for the verify_guess and inf_est functions."""

    def test_verify_guess_returns_0_or_1(
        self, spark, toy_graph, broadcast_factory
    ):
        """verify_guess should always return either 0 or 1."""
        bc = broadcast_factory(toy_graph)
        result = influence.verify_guess(
            spark, bc, s=["A"], n=3, tau=1, epsilon=influence.EPSILON, max_epochs=3
        )
        assert result in (0, 1)

    def test_inf_est_at_least_size_of_seed_set(
        self, spark, toy_graph, broadcast_factory
    ):
        """inf_est should never return a score smaller than the seed set size."""
        bc = broadcast_factory(toy_graph)
        score = influence.inf_est(spark, bc, s=["A"], n=3, epsilon=influence.EPSILON, max_epochs=3)
        assert score >= 1
