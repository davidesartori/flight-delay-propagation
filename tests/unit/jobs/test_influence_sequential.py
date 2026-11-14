"""Testing module for sequential influence job"""
import random
import pytest
from src.jobs import influence_sequential


@pytest.fixture
def toy_graph():
    """Toy graph."""
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


class TestSimulate:
    """Tests for the simulate function."""

    def test_returns_true_if_seed_set_already_reaches_t(self, toy_graph):
        """If the seeds alone are >= t, no cascade is needed."""
        assert influence_sequential.simulate(toy_graph, ["A", "B"], t=2) is True

    def test_always_infects_when_random_below_probability(
        self, monkeypatch, toy_graph
    ):
        """With random() == 0 every edge fires, so A reaches A, B, C."""
        monkeypatch.setattr(random, "random", lambda: 0.0)
        assert influence_sequential.simulate(toy_graph, ["A"], t=3) is True

    def test_never_infects_when_random_above_probability(
        self, monkeypatch, toy_graph
    ):
        """With random() == 1 no edge fires, so the cascade stops at the seed."""
        monkeypatch.setattr(random, "random", lambda: 1)
        assert influence_sequential.simulate(toy_graph, ["A"], t=2) is False

    def test_isolated_node_cannot_reach_t_greater_than_1(self):
        """A node with no outgoing edges cannot spread."""
        assert influence_sequential.simulate({"Z": []}, ["Z"], t=2) is False

    def test_node_missing_from_graph_is_handled(self):
        """Nodes absent from the graph are treated as having no neighbors."""
        assert influence_sequential.simulate({}, ["Q"], t=2) is False

    def test_respects_max_neighbors_t(self):
        """Only the first t neighbors of a node are considered."""
        graph = {"A": [("B", 0.0), ("C", 0.0), ("D", 1.0), ("E", 1.0)]}
        assert influence_sequential.simulate(graph, ["A"], t=2) is False

    def test_respects_max_epochs(self, monkeypatch, chain_graph):
        """A chain longer than MAX_EPOCHS cannot be fully traversed."""
        monkeypatch.setattr(random, "random", lambda: 0.0)
        monkeypatch.setattr(influence_sequential, "MAX_EPOCHS", 1)
        assert influence_sequential.simulate(chain_graph, ["A"], t=4) is False

        monkeypatch.setattr(influence_sequential, "MAX_EPOCHS", 3)
        assert influence_sequential.simulate(chain_graph, ["A"], t=4) is True

    def test_does_not_reinfect_already_infected_nodes(self, monkeypatch):
        """Cycles must not cause nodes to be counted twice."""
        monkeypatch.setattr(random, "random", lambda: 0.0)
        graph = {"A": [("B", 1.0)], "B": [("A", 1.0)]}
        assert influence_sequential.simulate(graph, ["A"], t=3) is False


class TestSampleOracle:
    """Tests for the sample_oracle function."""

    def test_returns_fraction_between_0_and_1(self, toy_graph):
        """sample_oracle should return a fraction in [0, 1]."""
        result = influence_sequential.sample_oracle(toy_graph, ["A"], l=20, t=2)
        assert 0.0 <= result <= 1.0

    def test_returns_one_when_cascade_always_succeeds(self, chain_graph):
        """Deterministic edges -> sample reaches the threshold."""
        result = influence_sequential.sample_oracle(chain_graph, ["A"], l=10, t=3)
        assert result == pytest.approx(1.0)

    def test_isolated_node_never_exceeds_threshold_gt_1(self):
        """An isolated seed never reaches a threshold greater than 1."""
        result = influence_sequential.sample_oracle({"Z": []}, ["Z"], l=10, t=2)
        assert result == pytest.approx(0.0)

    def test_calls_simulate_l_times(self, monkeypatch, toy_graph):
        """sample_oracle should run exactly l simulations."""
        calls = []

        def fake_simulate(graph, starting_nodes, t):
            calls.append(1)
            return True

        monkeypatch.setattr(influence_sequential, "simulate", fake_simulate)
        influence_sequential.sample_oracle(toy_graph, ["A"], l=7, t=2)
        assert len(calls) == 7

    def test_fraction_matches_simulate_outcomes(self, monkeypatch, toy_graph):
        """The fraction equals hits / l given a scripted sequence of outcomes."""
        outcomes = iter([True, False, True, False])
        monkeypatch.setattr(
            influence_sequential, "simulate", lambda *a, **k: next(outcomes)
        )
        result = influence_sequential.sample_oracle(toy_graph, ["A"], l=4, t=2)
        assert result == pytest.approx(0.5)


class TestVerifyGuessAndInfEst:
    """Tests for the verify_guess and inf_est functions."""

    def test_verify_guess_returns_bool(self, toy_graph):
        """verify_guess should always return a boolean."""
        result = influence_sequential.verify_guess(
            toy_graph, ["A"], tau=1, epsilon=influence_sequential.EPSILON
        )
        assert result in (True, False)

    def test_verify_guess_accepts_when_oracle_always_one(
        self, monkeypatch, toy_graph
    ):
        """If every threshold is always reached, the guess is accepted."""
        monkeypatch.setattr(influence_sequential, "sample_oracle", lambda *a, **k: 1.0)
        assert influence_sequential.verify_guess(
            toy_graph, ["A"], tau=1, epsilon=influence_sequential.EPSILON
        ) is True

    def test_verify_guess_rejects_when_oracle_always_zero(
        self, monkeypatch, toy_graph
    ):
        """If no threshold is ever reached, the guess is rejected."""
        monkeypatch.setattr(influence_sequential, "sample_oracle", lambda *a, **k: 0.0)
        assert influence_sequential.verify_guess(
            toy_graph, ["A"], tau=1, epsilon=influence_sequential.EPSILON
        ) is False

    def test_inf_est_at_least_size_of_seed_set(self, toy_graph):
        """inf_est should never return a score smaller than 1."""
        score = influence_sequential.inf_est(
            toy_graph, ["A"], epsilon=influence_sequential.EPSILON
        )
        assert score >= 1

    def test_inf_est_returns_n_when_first_guess_accepted(
        self, monkeypatch, toy_graph
    ):
        """If verify_guess accepts immediately, inf_est returns tau = n."""
        monkeypatch.setattr(influence_sequential, "verify_guess", lambda *a, **k: True)
        score = influence_sequential.inf_est(toy_graph, ["A"], epsilon=0.1)
        assert score == len(toy_graph)

    def test_inf_est_returns_1_when_all_guesses_rejected(
        self, monkeypatch, toy_graph
    ):
        """If every guess is rejected, the fallback value is 1."""
        monkeypatch.setattr(influence_sequential, "verify_guess", lambda *a, **k: False)
        score = influence_sequential.inf_est(toy_graph, ["A"], epsilon=0.1)
        assert score == 1

    def test_inf_est_decreases_tau_until_accepted(self, monkeypatch, toy_graph):
        """tau is reduced by a factor (1 + epsilon) after each rejection."""
        seen = []

        def fake_verify(graph, starting_nodes, tau, epsilon, n_samples=None):
            seen.append(tau)
            return len(seen) == 3  # accept on the third guess

        monkeypatch.setattr(influence_sequential, "verify_guess", fake_verify)
        score = influence_sequential.inf_est(toy_graph, ["A"], epsilon=0.1)

        assert len(seen) == 3
        assert seen[1] == pytest.approx(seen[0] / (1 + influence_sequential.EPSILON))
        assert score == pytest.approx(seen[2])
