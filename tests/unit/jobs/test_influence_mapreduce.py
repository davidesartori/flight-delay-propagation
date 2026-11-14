"""Testing module for MapReduce influence job"""
import math
import random
from types import SimpleNamespace

import pytest
from src.jobs import influence_mapreduce
from src.influence.common import NEW, OLD


@pytest.fixture
def toy_graph():
    """Small, Spark-independent toy graph."""
    return {
        "A": [("B", 0.9), ("C", 0.1)],
        "B": [("C", 0.5)],
        "C": [],
    }


@pytest.fixture
def chain_graph():
    """Deterministic chain A -> B -> C -> D (all probabilities 1.0)."""
    return {
        "A": [("B", 1.0)],
        "B": [("C", 1.0)],
        "C": [("D", 1.0)],
        "D": [],
    }


@pytest.fixture
def broadcast_factory(spark):
    """Creates a real Spark broadcast for a given graph."""
    def _make(graph):
        return spark.sparkContext.broadcast(graph)
    return _make


def fake_bc(graph):
    """Minimal stand-in for a Broadcast: the code only accesses `.value`."""
    return SimpleNamespace(value=graph)


class TestInfect:
    """Tests for the infect function."""

    def test_always_infects_when_random_below_probability(
        self, monkeypatch, toy_graph, broadcast_factory
    ):
        """All neighbors get infected when random() is below their probability."""
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
        """No neighbors get infected when random() is above their probability."""
        monkeypatch.setattr(random, "random", lambda: 1)
        bc = broadcast_factory(toy_graph)
        current = (("sample0", "A"), NEW)
        results = influence_mapreduce.infect(current, bc, t=10)

        assert results == [(("sample0", "A"), OLD)]

    def test_respects_max_neighbors_t(self, monkeypatch, broadcast_factory):
        """Only the first t neighbors are considered for infection."""
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
        """A node with no outgoing edges only returns itself, marked as old."""
        bc = broadcast_factory({})
        current = (("sample0", "Z"), NEW)
        results = influence_mapreduce.infect(current, bc, t=5)
        assert results == [(("sample0", "Z"), OLD)]

    def test_zero_probability_edge_never_fires(
        self, monkeypatch, broadcast_factory
    ):
        """random() is in [0, 1), so an edge with probability 0 never fires."""
        monkeypatch.setattr(random, "random", lambda: 0.0)
        bc = broadcast_factory({"A": [("B", 0.0)]})
        results = influence_mapreduce.infect((("sample0", "A"), NEW), bc, t=5)
        assert results == [(("sample0", "A"), OLD)]


class TestSampleOracle:
    """Tests for the sample_oracle function."""

    def test_returns_fraction_between_0_and_1(
        self, spark, toy_graph, broadcast_factory
    ):
        """sample_oracle should return a fraction in [0, 1]."""
        bc = broadcast_factory(toy_graph)
        result = influence_mapreduce.sample_oracle(
            spark, bc, ["A"], l=5, t=2, max_epochs=3
        )
        assert 0.0 <= result <= 1.0

    def test_isolated_node_never_exceeds_threshold_gt_1(
        self, spark, broadcast_factory
    ):
        """An isolated seed node never reaches a threshold greater than 1."""
        bc = broadcast_factory({"Z": []})
        result = influence_mapreduce.sample_oracle(
            spark, bc, ["Z"], l=3, t=2, max_epochs=3
        )
        assert result == pytest.approx(0.0)

    def test_returns_one_when_cascade_always_succeeds(
        self, spark, chain_graph, broadcast_factory
    ):
        """Deterministic edges -> every sample reaches the threshold."""
        bc = broadcast_factory(chain_graph)
        result = influence_mapreduce.sample_oracle(
            spark, bc, ["A"], l=5, t=3, max_epochs=3
        )
        assert result == pytest.approx(1.0)

    def test_respects_max_epochs(self, spark, chain_graph, broadcast_factory):
        """Reaching 4 nodes on the chain takes exactly 3 epochs."""
        bc = broadcast_factory(chain_graph)
        assert influence_mapreduce.sample_oracle(
            spark, bc, ["A"], l=5, t=4, max_epochs=2
        ) == pytest.approx(0.0)
        assert influence_mapreduce.sample_oracle(
            spark, bc, ["A"], l=5, t=4, max_epochs=3
        ) == pytest.approx(1.0)


    def test_threshold_1_is_always_reached(self, spark, broadcast_factory):
        """The seed alone is enough for t=1, even if isolated."""
        bc = broadcast_factory({"Z": []})
        result = influence_mapreduce.sample_oracle(
            spark, bc, ["Z"], l=5, t=1, max_epochs=3
        )
        assert result == pytest.approx(1.0)

    def test_seed_set_counts_as_infected(self, spark, broadcast_factory):
        """Two isolated seeds reach t=2 on their own; a single one does not."""
        bc = broadcast_factory({"Y": [], "Z": []})
        assert influence_mapreduce.sample_oracle(
            spark, bc, ["Y", "Z"], l=5, t=2, max_epochs=3
        ) == pytest.approx(1.0)
        assert influence_mapreduce.sample_oracle(
            spark, bc, ["Z"], l=5, t=2, max_epochs=3
        ) == pytest.approx(0.0)


class TestVerifyGuess:
    """Tests for the verify_guess function."""

    def test_accepts_when_oracle_always_one(self, monkeypatch, toy_graph):
        """If every threshold is always reached, the guess is accepted."""
        monkeypatch.setattr(
            influence_mapreduce, "sample_oracle", lambda *a, **k: 1.0
        )
        assert influence_mapreduce.verify_guess(
            None, fake_bc(toy_graph), ["A"], n=3, tau=1,
            epsilon=influence_mapreduce.EPSILON, max_epochs=3
        ) is True

    def test_rejects_when_oracle_always_zero(self, monkeypatch, toy_graph):
        """If no threshold is ever reached, the guess is rejected."""
        monkeypatch.setattr(
            influence_mapreduce, "sample_oracle", lambda *a, **k: 0.0
        )
        assert influence_mapreduce.verify_guess(
            None, fake_bc(toy_graph), ["A"], n=3, tau=1,
            epsilon=influence_mapreduce.EPSILON, max_epochs=3
        ) is False


    def test_n_samples_overrides_l(self, monkeypatch, toy_graph):
        """When n_samples is given, every oracle call uses l = n_samples."""
        ls = []
        monkeypatch.setattr(
            influence_mapreduce, "sample_oracle",
            lambda sp, bc, s, l, t, me: ls.append(l) or 0.0,
        )
        influence_mapreduce.verify_guess(
            None, fake_bc(toy_graph), ["A"], n=3, tau=1, epsilon=0.1,
            max_epochs=3, n_samples=7
        )
        assert ls and set(ls) == {7}

    def test_forwards_max_epochs(self, monkeypatch, toy_graph):
        """max_epochs is passed through to the oracle."""
        seen = []
        monkeypatch.setattr(
            influence_mapreduce, "sample_oracle",
            lambda sp, bc, s, l, t, me: seen.append(me) or 0.0,
        )
        influence_mapreduce.verify_guess(
            None, fake_bc(toy_graph), ["A"], n=3, tau=1, epsilon=0.1,
            max_epochs=5
        )
        assert set(seen) == {5}

    def test_no_oracle_call_when_tau_exceeds_n(self, monkeypatch, toy_graph):
        """If tau > n the loop never runs and the guess is rejected."""
        called = []
        monkeypatch.setattr(
            influence_mapreduce, "sample_oracle",
            lambda *a, **k: called.append(1) or 1.0,
        )
        assert influence_mapreduce.verify_guess(
            None, fake_bc(toy_graph), ["A"], n=3, tau=4, epsilon=0.1,
            max_epochs=3
        ) is False
        assert not called

    def test_real_spark_returns_bool(self, spark, toy_graph, broadcast_factory):
        """End-to-end with a small fixed number of samples."""
        bc = broadcast_factory(toy_graph)
        result = influence_mapreduce.verify_guess(
            spark, bc, ["A"], n=3, tau=1,
            epsilon=influence_mapreduce.EPSILON, max_epochs=3, n_samples=10
        )
        assert isinstance(result, bool)


class TestInfEst:
    """Tests for the inf_est function."""

    def test_returns_n_when_first_guess_accepted(self, monkeypatch, toy_graph):
        """If verify_guess accepts immediately, inf_est returns tau = n."""
        monkeypatch.setattr(
            influence_mapreduce, "verify_guess", lambda *a, **k: True
        )
        assert influence_mapreduce.inf_est(
            None, fake_bc(toy_graph), ["A"], n=3, epsilon=0.1, max_epochs=3
        ) == 3

    def test_returns_1_when_all_guesses_rejected(self, monkeypatch, toy_graph):
        """If every guess is rejected, the fallback value is 1."""
        calls = []
        monkeypatch.setattr(
            influence_mapreduce, "verify_guess",
            lambda sp, bc, s, n, tau, eps, me, ns=None: calls.append(tau) or False,
        )
        score = influence_mapreduce.inf_est(
            None, fake_bc(toy_graph), ["A"], n=3, epsilon=0.1, max_epochs=3
        )
        assert score == 1
        assert len(calls) == 12
        assert all(tau >= 1 for tau in calls)

    def test_decreases_tau_until_accepted(self, monkeypatch, toy_graph):
        """tau is divided by (1 + epsilon) after each rejection."""
        seen = []

        def fake_verify(sp, bc, s, n, tau, eps, me, ns=None):
            seen.append(tau)
            return len(seen) == 3

        monkeypatch.setattr(influence_mapreduce, "verify_guess", fake_verify)
        score = influence_mapreduce.inf_est(
            None, fake_bc(toy_graph), ["A"], n=3, epsilon=0.1, max_epochs=3
        )
        assert seen[0] == 3
        assert seen[1] == pytest.approx(3 / (1 + influence_mapreduce.EPSILON))
        assert score == pytest.approx(seen[2])

    def test_seed_set_larger_than_n_returns_1(self, monkeypatch, toy_graph):
        """If the seed set exceeds n, the loop is skipped and 1 is returned."""
        called = []
        monkeypatch.setattr(
            influence_mapreduce, "verify_guess",
            lambda *a, **k: called.append(1) or True,
        )
        assert influence_mapreduce.inf_est(
            None, fake_bc(toy_graph), ["A", "B", "C", "D"], n=3,
            epsilon=0.1, max_epochs=3
        ) == 1
        assert not called

    def test_forwards_parameters(self, monkeypatch, toy_graph):
        """n, epsilon, max_epochs and n_samples reach verify_guess."""
        seen = []

        def fake_verify(sp, bc, s, n, tau, eps, me, ns=None):
            seen.append((n, eps, me, ns))
            return True

        monkeypatch.setattr(influence_mapreduce, "verify_guess", fake_verify)
        influence_mapreduce.inf_est(
            None, fake_bc(toy_graph), ["A"], n=3, epsilon=0.2,
            max_epochs=4, n_samples=9
        )
        assert seen == [(3, 0.2, 4, 9)]

    def test_real_spark_score_in_range(self, spark, toy_graph, broadcast_factory):
        """The score is between 1 and n on a real (small) run."""
        bc = broadcast_factory(toy_graph)
        score = influence_mapreduce.inf_est(
            spark, bc, ["A"], n=3, epsilon=influence_mapreduce.EPSILON,
            max_epochs=3, n_samples=10
        )
        assert 1 <= score <= 3

    def test_real_spark_isolated_node(self, spark, broadcast_factory):
        """One-node graph with n_samples: a single cheap Spark oracle call."""
        bc = broadcast_factory({"Z": []})
        assert influence_mapreduce.inf_est(
            spark, bc, ["Z"], n=1, epsilon=0.1, max_epochs=3, n_samples=5
        ) == 1
