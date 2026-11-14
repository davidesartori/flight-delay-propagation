"""Testing module for batched MapReduce influence job"""
import math
import random
from types import SimpleNamespace

import pytest
from src.jobs import influence_mapreduce_batched as inf
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
    """Minimal stand-in for a Broadcast: the code only accesses 'value'"""
    return SimpleNamespace(value=graph)


class TestInfectBatched:
    """Tests for the infect_batched function."""

    def test_always_infects_when_random_below_probability(
        self, monkeypatch, toy_graph
    ):
        """All neighbors get infected when random() is below their probability."""
        monkeypatch.setattr(random, "random", lambda: 0.0)
        row = (("MXP", 0, "A"), NEW)
        results = inf.infect_batched(row, fake_bc(toy_graph), t=10)

        new_nodes = {node for node, label in results if label == NEW}
        old_nodes = {node for node, label in results if label == OLD}

        assert new_nodes == {("MXP", 0, "B"), ("MXP", 0, "C")}
        assert old_nodes == {("MXP", 0, "A")}

    def test_never_infects_when_random_above_probability(
        self, monkeypatch, toy_graph
    ):
        """No neighbors get infected when random() is above their probability."""
        monkeypatch.setattr(random, "random", lambda: 1)
        row = (("MXP", 0, "A"), NEW)
        results = inf.infect_batched(row, fake_bc(toy_graph), t=10)
        assert results == [(("MXP", 0, "A"), OLD)]

    def test_respects_max_neighbors_t(self, monkeypatch):
        """Only the first t neighbors are considered for infection."""
        monkeypatch.setattr(random, "random", lambda: 0.0)
        graph = {"A": [("B", 0.9), ("C", 0.8), ("D", 0.7)]}
        row = (("MXP", 0, "A"), NEW)

        results = inf.infect_batched(row, fake_bc(graph), t=2)
        new_nodes = {node for node, label in results if label == NEW}
        assert new_nodes == {("MXP", 0, "B"), ("MXP", 0, "C")}

    def test_node_with_no_outgoing_edges_only_returns_old_self(self):
        """A node with no outgoing edges only returns itself, marked as old."""
        row = (("MXP", 0, "Z"), NEW)
        results = inf.infect_batched(row, fake_bc({"Z": []}), t=5)
        assert results == [(("MXP", 0, "Z"), OLD)]

    def test_node_missing_from_graph_is_handled(self):
        """Nodes absent from the graph are treated as having no neighbors."""
        row = (("MXP", 0, "Q"), NEW)
        results = inf.infect_batched(row, fake_bc({}), t=5)
        assert results == [(("MXP", 0, "Q"), OLD)]

    def test_keys_keep_airport_and_sample_id(self, monkeypatch, toy_graph):
        """Every output key carries the same (airport, sample_id) as the input."""
        monkeypatch.setattr(random, "random", lambda: 0.0)
        row = (("JFK", 42, "A"), NEW)
        results = inf.infect_batched(row, fake_bc(toy_graph), t=10)
        assert all(key[0] == "JFK" and key[1] == 42 for key, _ in results)


class TestSampleOracleBatched:
    """Tests for sample_oracle_batched running on a real SparkSession."""

    def test_returns_fraction_between_0_and_1(
        self, spark, toy_graph, broadcast_factory
    ):
        """Every airport gets a fraction in [0, 1]."""
        bc = broadcast_factory(toy_graph)
        result = inf.sample_oracle_batched(
            spark, bc, ["A", "B"], l=10, t=2, max_epochs=3
        )
        assert set(result) == {"A", "B"}
        assert all(0.0 <= v <= 1.0 for v in result.values())

    def test_returns_one_when_cascade_always_succeeds(
        self, spark, chain_graph, broadcast_factory
    ):
        """Deterministic edges -> every sample reaches the threshold."""
        bc = broadcast_factory(chain_graph)
        result = inf.sample_oracle_batched(
            spark, bc, ["A"], l=5, t=3, max_epochs=3
        )
        assert result["A"] == pytest.approx(1.0)

    def test_isolated_node_never_exceeds_threshold_gt_1(
        self, spark, broadcast_factory
    ):
        """An isolated seed never reaches a threshold greater than 1."""
        bc = broadcast_factory({"Z": []})
        result = inf.sample_oracle_batched(
            spark, bc, ["Z"], l=5, t=2, max_epochs=3
        )
        assert result["Z"] == pytest.approx(0.0)

    def test_zero_probability_edges_never_spread(
        self, spark, broadcast_factory
    ):
        """Edges with probability 0 never fire (random() is in [0, 1))."""
        bc = broadcast_factory({"A": [("B", 0.0)], "B": []})
        result = inf.sample_oracle_batched(
            spark, bc, ["A"], l=5, t=2, max_epochs=3
        )
        assert result["A"] == pytest.approx(0.0)

    def test_threshold_1_is_always_reached(
        self, spark, broadcast_factory
    ):
        """With t=1 the seed alone is enough, even for an isolated node."""
        bc = broadcast_factory({"Z": []})
        result = inf.sample_oracle_batched(
            spark, bc, ["Z"], l=5, t=1, max_epochs=3
        )
        assert result["Z"] == pytest.approx(1.0)

    def test_respects_max_epochs(
        self, spark, chain_graph, broadcast_factory
    ):
        """Reaching 4 nodes on the chain takes exactly 3 epochs."""
        bc = broadcast_factory(chain_graph)

        too_few = inf.sample_oracle_batched(
            spark, bc, ["A"], l=5, t=4, max_epochs=2
        )
        assert too_few["A"] == pytest.approx(0.0)

        enough = inf.sample_oracle_batched(
            spark, bc, ["A"], l=5, t=4, max_epochs=3
        )
        assert enough["A"] == pytest.approx(1.0)

    def test_airports_are_processed_independently(
        self, spark, chain_graph, broadcast_factory
    ):
        """Results of different airports in the same batch do not mix."""
        bc = broadcast_factory(chain_graph)
        result = inf.sample_oracle_batched(
            spark, bc, ["A", "D"], l=5, t=3, max_epochs=3
        )
        assert result["A"] == pytest.approx(1.0)
        assert result["D"] == pytest.approx(0.0)


class TestVerifyGuessBatched:
    """Tests for the verify_guess_batched function."""

    def test_accepts_all_when_oracle_always_one(self, monkeypatch, toy_graph):
        """If every threshold is always reached, all guesses are accepted."""
        monkeypatch.setattr(
            inf, "sample_oracle_batched",
            lambda sp, bc, airports, l, t, me: {a: 1.0 for a in airports},
        )
        result = inf.verify_guess_batched(
            None, fake_bc(toy_graph), ["A", "B"], n=3, tau=1,
            epsilon=inf.EPSILON, max_epochs=3,
        )
        assert result == {"A": 1, "B": 1}

    def test_rejects_all_when_oracle_always_zero(self, monkeypatch, toy_graph):
        """If no threshold is ever reached, all guesses are rejected."""
        monkeypatch.setattr(
            inf, "sample_oracle_batched",
            lambda sp, bc, airports, l, t, me: {a: 0.0 for a in airports},
        )
        result = inf.verify_guess_batched(
            None, fake_bc(toy_graph), ["A", "B"], n=3, tau=1,
            epsilon=inf.EPSILON, max_epochs=3,
        )
        assert result == {"A": 0, "B": 0}

    def test_decided_airports_are_dropped_from_later_calls(
        self, monkeypatch, toy_graph
    ):
        """Once an airport is accepted it is no longer sent to the oracle."""
        calls = []

        def spy(sp, bc, airports, l, t, me):
            calls.append(set(airports))
            return {a: (1.0 if a == "A" else 0.0) for a in airports}

        monkeypatch.setattr(inf, "sample_oracle_batched", spy)
        result = inf.verify_guess_batched(
            None, fake_bc(toy_graph), ["A", "B"], n=3, tau=1,
            epsilon=inf.EPSILON, max_epochs=3,
        )

        assert result == {"A": 1, "B": 0}
        assert calls[0] == {"A", "B"}
        assert calls[-1] == {"B"}
        first_without_a = next(i for i, c in enumerate(calls) if "A" not in c)
        assert all("A" not in c for c in calls[first_without_a:])

    def test_stops_when_all_airports_decided(self, monkeypatch, toy_graph):
        """The t-loop ends early if nobody is left undecided."""
        calls = []

        def spy(sp, bc, airports, l, t, me):
            calls.append(t)
            return {a: 1.0 for a in airports}

        monkeypatch.setattr(inf, "sample_oracle_batched", spy)
        inf.verify_guess_batched(
            None, fake_bc(toy_graph), ["A"], n=3, tau=1,
            epsilon=inf.EPSILON, max_epochs=3,
        )

        assert len(calls) < 12

    def test_n_samples_overrides_l(self, monkeypatch, toy_graph):
        """When n_samples is given, every oracle call uses l = n_samples."""
        ls = []

        def spy(sp, bc, airports, l, t, me):
            ls.append(l)
            return {a: 0.0 for a in airports}

        monkeypatch.setattr(inf, "sample_oracle_batched", spy)
        inf.verify_guess_batched(
            None, fake_bc(toy_graph), ["A"], n=3, tau=1,
            epsilon=0.1, max_epochs=3, n_samples=7,
        )
        assert ls and set(ls) == {7}

    def test_forwards_max_epochs(self, monkeypatch, toy_graph):
        """max_epochs is passed through to the oracle."""
        seen = []

        def spy(sp, bc, airports, l, t, me):
            seen.append(me)
            return {a: 0.0 for a in airports}

        monkeypatch.setattr(inf, "sample_oracle_batched", spy)
        inf.verify_guess_batched(
            None, fake_bc(toy_graph), ["A"], n=3, tau=1,
            epsilon=0.1, max_epochs=5,
        )
        assert set(seen) == {5}

    def test_no_oracle_call_when_tau_exceeds_n(self, monkeypatch, toy_graph):
        """If tau > n the loop never runs and the guess is rejected."""
        called = []
        monkeypatch.setattr(
            inf, "sample_oracle_batched",
            lambda *a, **k: called.append(1) or {},
        )
        result = inf.verify_guess_batched(
            None, fake_bc(toy_graph), ["A"], n=3, tau=4,
            epsilon=0.1, max_epochs=3,
        )
        assert result == {"A": 0}
        assert not called

    def test_empty_airports_returns_empty_dict(self, monkeypatch, toy_graph):
        """No airports -> no oracle calls and an empty result."""
        called = []
        monkeypatch.setattr(
            inf, "sample_oracle_batched",
            lambda *a, **k: called.append(1) or {},
        )
        result = inf.verify_guess_batched(
            None, fake_bc(toy_graph), [], n=3, tau=1,
            epsilon=0.1, max_epochs=3,
        )
        assert result == {}
        assert not called

    def test_real_spark_rejects_high_tau(
        self, spark, chain_graph, broadcast_factory
    ):
        """tau = n is too ambitious for a 4-node chain: a single oracle call, rejected."""
        bc = broadcast_factory(chain_graph)
        result = inf.verify_guess_batched(
            spark, bc, ["A"], n=4, tau=4, epsilon=0.1,
            max_epochs=3, n_samples=5,
        )
        assert result == {"A": 0}


class TestInfEstBatched:
    """Tests for the inf_est_batched function."""

    def test_returns_n_when_first_guess_accepted(self, monkeypatch, toy_graph):
        """If everyone is accepted immediately, every score is tau = n."""
        monkeypatch.setattr(
            inf, "verify_guess_batched",
            lambda sp, bc, airports, n, tau, eps, me, ns=None: {a: 1 for a in airports},
        )
        result = inf.inf_est_batched(
            None, fake_bc(toy_graph), ["A", "B"], n=3, epsilon=0.1, max_epochs=3
        )
        assert result == {"A": 3, "B": 3}

    def test_falls_back_to_1_when_all_guesses_rejected(
        self, monkeypatch, toy_graph
    ):
        """If every guess is rejected, the fallback score is 1."""
        calls = []

        def fake_verify(sp, bc, airports, n, tau, eps, me, ns=None):
            calls.append(tau)
            return {a: 0 for a in airports}

        monkeypatch.setattr(inf, "verify_guess_batched", fake_verify)
        result = inf.inf_est_batched(
            None, fake_bc(toy_graph), ["A", "B"], n=3, epsilon=0.1, max_epochs=3
        )

        assert result == {"A": 1, "B": 1}
        assert len(calls) == 12
        assert all(tau >= 1 for tau in calls)

    def test_airports_accepted_at_different_taus(self, monkeypatch, toy_graph):
        """Each airport gets the tau at which it was accepted; others keep going."""
        calls = []

        def fake_verify(sp, bc, airports, n, tau, eps, me, ns=None):
            calls.append((tau, set(airports)))
            step = len(calls)
            return {
                a: 1 if (a == "A" and step == 1) or (a == "B" and step == 3) else 0
                for a in airports
            }

        monkeypatch.setattr(inf, "verify_guess_batched", fake_verify)
        result = inf.inf_est_batched(
            None, fake_bc(toy_graph), ["A", "B"], n=3, epsilon=0.1, max_epochs=3
        )

        assert result["A"] == 3
        assert result["B"] == pytest.approx(3 / 1.1 ** 2)
        assert [airports for _, airports in calls] == [{"A", "B"}, {"B"}, {"B"}]
        assert len(calls) == 3

    def test_mixed_accepted_and_fallback(self, monkeypatch, toy_graph):
        """Accepted airports keep their tau, rejected ones fall back to 1."""
        monkeypatch.setattr(
            inf, "verify_guess_batched",
            lambda sp, bc, airports, n, tau, eps, me, ns=None:
                {a: 1 if a == "A" else 0 for a in airports},
        )
        result = inf.inf_est_batched(
            None, fake_bc(toy_graph), ["A", "B"], n=3, epsilon=0.1, max_epochs=3
        )
        assert result == {"A": 3, "B": 1}

    def test_forwards_parameters(self, monkeypatch, toy_graph):
        """epsilon, max_epochs and n_samples reach verify_guess_batched."""
        seen = []

        def fake_verify(sp, bc, airports, n, tau, eps, me, ns=None):
            seen.append((n, eps, me, ns))
            return {a: 1 for a in airports}

        monkeypatch.setattr(inf, "verify_guess_batched", fake_verify)
        inf.inf_est_batched(
            None, fake_bc(toy_graph), ["A"], n=3, epsilon=0.2,
            max_epochs=4, n_samples=9,
        )
        assert seen == [(3, 0.2, 4, 9)]

    def test_empty_airports_returns_empty_dict(self, monkeypatch, toy_graph):
        """No airports -> no calls and an empty result."""
        called = []
        monkeypatch.setattr(
            inf, "verify_guess_batched",
            lambda *a, **k: called.append(1) or {},
        )
        result = inf.inf_est_batched(
            None, fake_bc(toy_graph), [], n=3, epsilon=0.1, max_epochs=3
        )
        assert not result
        assert not called

    def test_real_spark_smoke_isolated_node(
        self, spark, broadcast_factory
    ):
        """End-to-end on a 1-node graph"""
        bc = broadcast_factory({"Z": []})
        result = inf.inf_est_batched(
            spark, bc, ["Z"], n=1, epsilon=0.1, max_epochs=3, n_samples=5
        )
        assert result == {"Z": 1}
