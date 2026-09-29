"""Testing module for influence common functions"""
import pytest

from src.influence.common import (
    estimate_rdd_size_for_epoch,
    compute_avg_out_degree,
    merge_labels
)


class TestEstimateRddSizeForEpoch:
    """Tests for RDD size estimation."""

    def test_basic_case(self):
        """Test basic RDD size estimation."""
        result = estimate_rdd_size_for_epoch(
            n_active_samples=10,
            avg_out_degree=2,
            n_total_nodes=100,
            epoch=1,
        )

        assert result == 20

    def test_growth_with_epoch(self):
        """Test growth across epochs."""
        result = estimate_rdd_size_for_epoch(
            n_active_samples=10,
            avg_out_degree=3,
            n_total_nodes=1000,
            epoch=2,
        )

        assert result == 90

    def test_capped_by_total_nodes(self):
        """Test the node count cap."""
        result = estimate_rdd_size_for_epoch(
            n_active_samples=10,
            avg_out_degree=10,
            n_total_nodes=50,
            epoch=3,
        )

        assert result == 500

    def test_zero_active_samples(self):
        """Test zero active samples."""
        result = estimate_rdd_size_for_epoch(
            n_active_samples=0,
            avg_out_degree=10,
            n_total_nodes=100,
            epoch=3,
        )

        assert result == 0

    def test_empty_graph_parameters(self):
        """Test zero average out-degree."""
        result = estimate_rdd_size_for_epoch(
            n_active_samples=10,
            avg_out_degree=0,
            n_total_nodes=100,
            epoch=3,
        )

        assert result == 0


class TestComputeAvgOutDegree:
    """Tests for average out-degree computation."""

    def test_basic_case(self):
        """Test basic average out-degree."""
        graph = {
            "A": [("B", 0.5), ("C", 0.3)],
            "B": [("C", 0.4)],
            "C": [],
        }

        result = compute_avg_out_degree(graph)

        assert result == pytest.approx(1.0)

    def test_empty_graph(self):
        """Test empty graph."""
        result = compute_avg_out_degree({})

        assert result == 0.0

    def test_single_node(self):
        """Test a single node."""
        graph = {
            "A": [("B", 0.5), ("C", 0.3), ("D", 0.2)]
        }

        result = compute_avg_out_degree(graph)

        assert result == 3.0

    def test_nodes_without_neighbors(self):
        """Test nodes without neighbors."""
        graph = {
            "A": [],
            "B": [],
            "C": [("D", 0.5)],
        }

        result = compute_avg_out_degree(graph)

        assert result == pytest.approx(1 / 3)

    def test_multiple_nodes(self):
        """Test multiple nodes."""
        graph = {
            "A": [("B", 0.5), ("C", 0.5)],
            "B": [("C", 0.5), ("D", 0.5)],
            "C": [("A", 0.5)],
            "D": [],
        }

        result = compute_avg_out_degree(graph)

        assert result == pytest.approx(1.25)


class TestMergeLabels:
    """Tests for label merging."""

    @pytest.mark.parametrize("l1,l2,expected", [
        ("old", "old", "old"),
        ("old", "new", "old"),
        ("new", "old", "old"),
        ("new", "new", "new"),
    ])
    def test_merge_labels_truth_table(self, l1, l2, expected):
        """Test label merging priority."""
        assert merge_labels(l1, l2) == expected
