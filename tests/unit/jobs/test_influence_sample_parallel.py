"""Testing module for sample-parallel influence job"""
import math
import random
from types import SimpleNamespace

import pytest
from src.jobs import influence_sample_parallel as inf


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
def sc(spark):
    """Real SparkContext."""
    return spark.sparkContext


@pytest.fixture
def broadcast_factory(sc):
    """Creates a real Spark broadcast for a given graph."""
    def _make(graph):
        return sc.broadcast(graph)
    return _make


def fake_bc(graph):
    """Minimal stand-in for a Broadcast: the code only accesses 'value'"""
    return SimpleNamespace(value=graph)


class FakeRDD:
    """Runs map/sum locally on the driver."""

    def __init__(self, data):
        self.data = data
        self.func = None

    def map(self, func):
        self.func = func
        return self

    def sum(self):
        return sum(self.func(x) for x in self.data)


class FakeSC:
    """SparkContext stand-in that records parallelize calls and runs on the driver,
    so monkeypatching 'random` / 'simulate' works."""

    def __init__(self, default_parallelism=2):
        self.defaultParallelism = default_parallelism
        self.calls = []

    def parallelize(self, data, num_slices):
        data = list(data)
        self.calls.append((data, num_slices))
        return FakeRDD(data)

    @property
    def sizes(self):
        return self.calls[0][0]


class TestSimulate:
    """Tests for the simulate function."""

    def test_returns_true_if_seed_set_already_reaches_t(self, toy_graph):
        """If the seeds alone are >= t, no cascade is needed."""
        assert inf.simulate(toy_graph, ["A", "B"], t=2) is True

    def test_always_infects_when_random_below_probability(
        self, monkeypatch, toy_graph
    ):
        """With random() == 0 every edge fires, so A reaches A, B, C."""
        monkeypatch.setattr(random, "random", lambda: 0.0)
        assert inf.simulate(toy_graph, ["A"], t=3) is True

    def test_never_infects_when_random_above_probability(
        self, monkeypatch, toy_graph
    ):
        """With random() == 1 no edge fires, so the cascade stops at the seed."""
        monkeypatch.setattr(random, "random", lambda: 1)
        assert inf.simulate(toy_graph, ["A"], t=2) is False

    def test_isolated_node_cannot_reach_t_greater_than_1(self):
        """A node with no outgoing edges cannot spread."""
        assert inf.simulate({"Z": []}, ["Z"], t=2) is False

    def test_node_missing_from_graph_is_handled(self):
        """Nodes absent from the graph are treated as having no neighbors."""
        assert inf.simulate({}, ["Q"], t=2) is False

    def test_respects_max_neighbors_t(self, monkeypatch):
        """Only the first t neighbors of a node are examined."""
        calls = []

        def counting_random():
            calls.append(1)
            return 0.0

        monkeypatch.setattr(random, "random", counting_random)
        graph = {"A": [("B", 1.0), ("C", 1.0), ("D", 1.0), ("E", 1.0)]}

        inf.simulate(graph, ["A"], t=2)
        assert len(calls) == 2  # only B and C were considered

    def test_respects_max_epochs(self, monkeypatch, chain_graph):
        """A chain longer than MAX_EPOCHS cannot be fully traversed."""
        monkeypatch.setattr(random, "random", lambda: 0.0)

        monkeypatch.setattr(inf, "MAX_EPOCHS", 1)
        assert inf.simulate(chain_graph, ["A"], t=4) is False

        monkeypatch.setattr(inf, "MAX_EPOCHS", 3)
        assert inf.simulate(chain_graph, ["A"], t=4) is True

    def test_does_not_reinfect_already_infected_nodes(self, monkeypatch):
        """Cycles must not cause nodes to be counted twice."""
        monkeypatch.setattr(random, "random", lambda: 0.0)
        graph = {"A": [("B", 1.0)], "B": [("A", 1.0)]}
        assert inf.simulate(graph, ["A"], t=3) is False


class TestRunBlock:
    """Tests for the run_block function."""

    def test_counts_all_hits_on_deterministic_graph(self, chain_graph):
        """Every sample reaches t on a deterministic chain."""
        assert inf.run_block(5, fake_bc(chain_graph), ["A"], t=3) == 5

    def test_counts_zero_hits_on_isolated_node(self):
        """No sample reaches t > 1 from an isolated node."""
        assert inf.run_block(5, fake_bc({"Z": []}), ["Z"], t=2) == 0

    def test_zero_size_returns_zero(self, toy_graph):
        """An empty block produces no hits."""
        assert inf.run_block(0, fake_bc(toy_graph), ["A"], t=2) == 0

    def test_runs_exactly_size_simulations(self, monkeypatch, toy_graph):
        """run_block should call simulate `size` times, using the broadcast value."""
        seen = []

        def fake_simulate(graph, starting_nodes, t):
            seen.append(graph)
            return True

        monkeypatch.setattr(inf, "simulate", fake_simulate)
        hits = inf.run_block(7, fake_bc(toy_graph), ["A"], t=2)

        assert hits == 7
        assert len(seen) == 7
        assert all(g is toy_graph for g in seen)


class TestSampleOracleBlocks:
    """Tests for how sample_oracle splits l samples into blocks."""

    def test_small_l_uses_single_block(self, toy_graph):
        """With l < MIN_BLOCK there is one block of size l."""
        fsc = FakeSC(default_parallelism=2)
        inf.sample_oracle(fsc, fake_bc(toy_graph), ["A"], l=10, t=2)
        assert fsc.sizes == [10]

    def test_uses_min_block_when_larger_than_block_max(self, toy_graph):
        """l=1200, parallelism=2 -> block_max=300 < MIN_BLOCK=500."""
        fsc = FakeSC(default_parallelism=2)
        inf.sample_oracle(fsc, fake_bc(toy_graph), ["A"], l=1200, t=2)
        assert fsc.sizes == [500, 500, 200]

    def test_uses_block_max_when_larger_than_min_block(self, toy_graph):
        """l=10000, parallelism=2 -> 4 tasks of 2500 samples each."""
        fsc = FakeSC(default_parallelism=2)
        inf.sample_oracle(fsc, fake_bc(toy_graph), ["A"], l=10000, t=2)
        assert fsc.sizes == [2500] * 4

    def test_explicit_block_size_overrides_default(self, toy_graph):
        """block_size takes precedence over MIN_BLOCK / block_max."""
        fsc = FakeSC(default_parallelism=2)
        inf.sample_oracle(fsc, fake_bc(toy_graph), ["A"], l=250, t=2, block_size=100)
        assert fsc.sizes == [100, 100, 50]

    @pytest.mark.parametrize("l", [1, 10, 499, 500, 501, 1234, 10000])
    @pytest.mark.parametrize("block_size", [None, 7, 100])
    def test_block_sizes_sum_to_l(self, toy_graph, l, block_size):
        """Whatever the split, exactly l samples are drawn."""
        fsc = FakeSC(default_parallelism=3)
        inf.sample_oracle(
            fsc, fake_bc(toy_graph), ["A"], l=l, t=2, block_size=block_size
        )
        assert sum(fsc.sizes) == l
        assert all(s > 0 for s in fsc.sizes)

    def test_number_of_partitions_equals_number_of_blocks(self, toy_graph):
        """parallelize is called with one slice per block."""
        fsc = FakeSC(default_parallelism=2)
        inf.sample_oracle(fsc, fake_bc(toy_graph), ["A"], l=1200, t=2)
        data, num_slices = fsc.calls[0]
        assert num_slices == len(data)

    def test_fraction_matches_simulate_outcomes(self, monkeypatch, toy_graph):
        """The fraction equals hits / l given scripted simulate outcomes."""
        outcomes = iter([True, False, True, False])
        monkeypatch.setattr(inf, "simulate", lambda *a, **k: next(outcomes))
        fsc = FakeSC()
        result = inf.sample_oracle(fsc, fake_bc(toy_graph), ["A"], l=4, t=2)
        assert result == pytest.approx(0.5)


class TestSampleOracleSpark:
    """Tests for sample_oracle running on a real SparkContext."""

    def test_returns_fraction_between_0_and_1(
        self, sc, toy_graph, broadcast_factory
    ):
        """sample_oracle should return a fraction in [0, 1]."""
        bc = broadcast_factory(toy_graph)
        result = inf.sample_oracle(sc, bc, ["A"], l=50, t=2)
        assert 0.0 <= result <= 1.0

    def test_returns_one_when_cascade_always_succeeds(
        self, sc, chain_graph, broadcast_factory
    ):
        """Deterministic edges -> every sample reaches the threshold."""
        bc = broadcast_factory(chain_graph)
        result = inf.sample_oracle(sc, bc, ["A"], l=20, t=3)
        assert result == pytest.approx(1.0)

    def test_isolated_node_never_exceeds_threshold_gt_1(
        self, sc, broadcast_factory
    ):
        """An isolated seed never reaches a threshold greater than 1."""
        bc = broadcast_factory({"Z": []})
        result = inf.sample_oracle(sc, bc, ["Z"], l=20, t=2)
        assert result == pytest.approx(0.0)

    def test_multiple_blocks_aggregate_correctly(
        self, sc, chain_graph, broadcast_factory
    ):
        """With several small blocks the hits are still summed over all of them."""
        bc = broadcast_factory(chain_graph)
        result = inf.sample_oracle(sc, bc, ["A"], l=25, t=3, block_size=10)
        assert result == pytest.approx(1.0)


class TestVerifyGuess:
    """Tests for the verify_guess function."""

    def test_accepts_when_oracle_always_one(self, monkeypatch, toy_graph):
        """If every threshold is always reached, the guess is accepted."""
        monkeypatch.setattr(inf, "sample_oracle", lambda *a, **k: 1.0)
        assert inf.verify_guess(
            None, fake_bc(toy_graph), ["A"], tau=1, epsilon=inf.EPSILON
        ) is True

    def test_rejects_when_oracle_always_zero(self, monkeypatch, toy_graph):
        """If no threshold is ever reached, the guess is rejected."""
        monkeypatch.setattr(inf, "sample_oracle", lambda *a, **k: 0.0)
        assert inf.verify_guess(
            None, fake_bc(toy_graph), ["A"], tau=1, epsilon=inf.EPSILON
        ) is False

    def test_first_call_uses_expected_l(self, monkeypatch, toy_graph):
        """l = max(10, ceil(8 * t * log(n)^3 / (eps^2 * tau))) at t = tau."""
        calls = []
        monkeypatch.setattr(
            inf, "sample_oracle",
            lambda sc, bc, s, l, t, *a, **k: calls.append(l) or 0.0,
        )
        eps, tau = 0.1, 2
        inf.verify_guess(None, fake_bc(toy_graph), ["A"], tau=tau, epsilon=eps)

        n = len(toy_graph)
        expected = max(10, math.ceil(8 * tau * (math.log(n) ** 3) / (eps ** 2 * tau)))
        assert calls[0] == expected

    def test_no_oracle_call_when_tau_exceeds_n(self, monkeypatch, toy_graph):
        """If tau > n the while loop never runs and the guess is rejected."""
        called = []
        monkeypatch.setattr(
            inf, "sample_oracle", lambda *a, **k: called.append(1) or 1.0
        )
        result = inf.verify_guess(
            None, fake_bc(toy_graph), ["A"], tau=len(toy_graph) + 1, epsilon=0.1
        )
        assert result is False
        assert not called


class TestInfEst:
    """Tests for the inf_est function."""

    def test_returns_n_when_first_guess_accepted(self, monkeypatch, toy_graph):
        """If verify_guess accepts immediately, inf_est returns tau = n."""
        monkeypatch.setattr(inf, "verify_guess", lambda *a, **k: True)
        score = inf.inf_est(None, fake_bc(toy_graph), ["A"], epsilon=0.1)
        assert score == len(toy_graph)

    def test_returns_1_when_all_guesses_rejected(self, monkeypatch, toy_graph):
        """If every guess is rejected, the fallback value is 1."""
        monkeypatch.setattr(inf, "verify_guess", lambda *a, **k: False)
        score = inf.inf_est(None, fake_bc(toy_graph), ["A"], epsilon=0.1)
        assert score == 1

    def test_decreases_tau_until_accepted(self, monkeypatch, toy_graph):
        """tau is divided by (1 + epsilon) after each rejection."""
        seen = []

        def fake_verify(sc, graph_bc, starting_nodes, tau, epsilon, n_samples=None):
            seen.append(tau)
            return len(seen) == 3  # accept on the third guess

        monkeypatch.setattr(inf, "verify_guess", fake_verify)
        score = inf.inf_est(None, fake_bc(toy_graph), ["A"], epsilon=0.1)

        assert len(seen) == 3
        assert seen[0] == len(toy_graph)
        assert seen[1] == pytest.approx(seen[0] / 1.1)
        assert score == pytest.approx(seen[2])

    def test_stops_when_tau_below_seed_set_size(self, monkeypatch, toy_graph):
        """No guess is attempted below len(starting_nodes)."""
        seen = []
        monkeypatch.setattr(
            inf, "verify_guess",
            lambda sc, bc, s, tau, eps, n_samples: seen.append(tau) or False,
        )
        inf.inf_est(None, fake_bc(toy_graph), ["A", "B"], epsilon=0.1)
        assert all(tau >= 2 for tau in seen)

    def test_seed_set_larger_than_graph_returns_1(self, monkeypatch, toy_graph):
        """If the seed set exceeds n, the loop is skipped and 1 is returned."""
        called = []
        monkeypatch.setattr(
            inf, "verify_guess", lambda *a, **k: called.append(1) or True
        )
        score = inf.inf_est(
            None, fake_bc(toy_graph), ["A", "B", "C", "D"], epsilon=0.1
        )
        assert score == 1
        assert not called
