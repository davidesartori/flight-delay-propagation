"""Tests for graph creation and label merging in influence common functions."""
import pytest

from src.influence.common import (
    merge_labels,
    create_graph,
    DELAY_THRESHOLD,
    NEW,
    OLD
)

COLS = ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DEPARTURE_DELAY"]


def make_df(spark, rows):
    """Create a Spark DataFrame from origin, destination, and delay rows."""
    return spark.createDataFrame(rows, schema=COLS)


def probs(graph, origin):
    """Return a dictionary mapping each destination to its probability for an origin."""
    return dict(graph[origin])


class TestCreateGraph:
    """Test the graph creation function and its route probabilities."""

    def test_probability_is_delayed_over_total(self, spark):
        """Verify that the probability equals delayed flights divided by total flights."""
        rows = [
            ("ATL", "ORD", DELAY_THRESHOLD + 10),
            ("ATL", "ORD", DELAY_THRESHOLD + 1),
            ("ATL", "ORD", 0),
            ("ATL", "ORD", -5),
        ]
        graph = create_graph(make_df(spark, rows))
        assert probs(graph, "ATL")["ORD"] == pytest.approx(0.5)

    def test_delay_exactly_at_threshold_is_not_delayed(self, spark):
        """Verify that a delay equal to the threshold is not counted as delayed."""
        rows = [
            ("ATL", "ORD", DELAY_THRESHOLD),
            ("ATL", "ORD", DELAY_THRESHOLD),
        ]
        graph = create_graph(make_df(spark, rows))
        assert probs(graph, "ATL")["ORD"] == pytest.approx(0.0)

    def test_all_delayed_gives_probability_one(self, spark):
        """Verify that the probability is one when all flights are delayed."""
        rows = [("ATL", "ORD", DELAY_THRESHOLD + 5)] * 3
        graph = create_graph(make_df(spark, rows))
        assert probs(graph, "ATL")["ORD"] == pytest.approx(1.0)

    def test_none_delayed_gives_probability_zero(self, spark):
        """Verify that the probability is zero when no flights are delayed."""
        rows = [("ATL", "ORD", 0)] * 3
        graph = create_graph(make_df(spark, rows))
        assert probs(graph, "ATL")["ORD"] == pytest.approx(0.0)

    def test_routes_are_aggregated_per_origin_destination_pair(self, spark):
        """Verify that each origin-destination pair produces a single edge."""
        rows = [
            ("ATL", "ORD", DELAY_THRESHOLD + 5),
            ("ATL", "ORD", 0),
            ("ATL", "DFW", DELAY_THRESHOLD + 5),
        ]
        graph = create_graph(make_df(spark, rows))
        assert len(graph["ATL"]) == 2
        assert probs(graph, "ATL")["ORD"] == pytest.approx(0.5)
        assert probs(graph, "ATL")["DFW"] == pytest.approx(1.0)

    def test_direction_matters(self, spark):
        """Verify that opposite routes have independent probabilities."""
        rows = [
            ("ATL", "ORD", DELAY_THRESHOLD + 5),
            ("ORD", "ATL", 0),
        ]
        graph = create_graph(make_df(spark, rows))
        assert probs(graph, "ATL")["ORD"] == pytest.approx(1.0)
        assert probs(graph, "ORD")["ATL"] == pytest.approx(0.0)

    def test_neighbors_sorted_by_probability_descending(self, spark):
        """Verify that outgoing neighbors are sorted by decreasing probability."""
        rows = (
            [("ATL", "AAA", 0)] * 4
            + [
                ("ATL", "BBB", DELAY_THRESHOLD + 5),
                ("ATL", "BBB", 0),
            ]
            + [("ATL", "CCC", DELAY_THRESHOLD + 5)] * 2
        )
        graph = create_graph(make_df(spark, rows))
        probabilities = [p for _, p in graph["ATL"]]
        destinations = [d for d, _ in graph["ATL"]]

        assert probabilities == sorted(probabilities, reverse=True)
        assert destinations == ["CCC", "BBB", "AAA"]

    def test_destination_only_airports_are_not_keys(self, spark):
        """Verify that airports appearing only as destinations are not graph keys."""
        rows = [("ATL", "ORD", 0)]
        graph = create_graph(make_df(spark, rows))
        assert "ATL" in graph
        assert "ORD" not in graph
        assert len(graph) == 1

    def test_null_delay_counts_as_flight_but_not_delayed(self, spark):
        """Verify that a null delay counts toward total flights but not delayed flights."""
        rows = [
            ("ATL", "ORD", DELAY_THRESHOLD + 5),
            ("ATL", "ORD", None),
        ]
        graph = create_graph(make_df(spark, rows))
        assert probs(graph, "ATL")["ORD"] == pytest.approx(0.5)

    def test_probabilities_are_in_unit_interval(self, spark):
        """Verify that every edge probability lies between zero and one."""
        rows = [
            ("ATL", "ORD", DELAY_THRESHOLD + 5),
            ("ATL", "ORD", 0),
            ("ATL", "DFW", 0),
            ("ORD", "ATL", DELAY_THRESHOLD + 5),
            ("ORD", "DFW", None),
        ]
        graph = create_graph(make_df(spark, rows))

        for neighbors in graph.values():
            for _, probability in neighbors:
                assert 0.0 <= probability <= 1.0

    def test_empty_dataframe_gives_empty_graph(self, spark):
        """Verify that an empty DataFrame produces an empty graph."""
        df = spark.createDataFrame(
            [],
            schema=(
                "ORIGIN_AIRPORT string, "
                "DESTINATION_AIRPORT string, "
                "DEPARTURE_DELAY double"
            ),
        )
        assert create_graph(df) == {}

    def test_each_edge_appears_once(self, spark):
        """Verify that duplicate flights do not create duplicate edges."""
        rows = (
            [("ATL", "ORD", 0)] * 5
            + [("ATL", "DFW", 0)] * 3
        )
        graph = create_graph(make_df(spark, rows))
        destinations = [d for d, _ in graph["ATL"]]
        assert len(destinations) == len(set(destinations)) == 2


class TestMergeLabels:
    """Test the merging rules for node labels."""

    @pytest.mark.parametrize(
        "label1,label2,expected",
        [
            (OLD, OLD, OLD),
            (OLD, NEW, OLD),
            (NEW, OLD, OLD),
            (NEW, NEW, NEW),
        ],
    )
    def test_merge_labels_truth_table(self, label1, label2, expected):
        """Verify label merging against all combinations of OLD and NEW."""
        assert merge_labels(label1, label2) == expected
