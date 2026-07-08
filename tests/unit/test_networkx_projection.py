from __future__ import annotations

from icframe.analytics.networkx import NetworkXAnalyzer
from icframe.domain.state import InteractionEdge, InteractionGraph


def test_collusion_metrics_on_fixed_graph_fixture() -> None:
    analyzer = NetworkXAnalyzer()
    graph = InteractionGraph(
        nodes=("alice", "bob", "cara"),
        edges=[
            InteractionEdge(source="alice", target="bob", weight=5.0, event_count=5),
            InteractionEdge(source="bob", target="alice", weight=5.0, event_count=5),
            InteractionEdge(source="alice", target="cara", weight=1.0, event_count=1),
        ],
    )

    metrics = analyzer.summarize(graph)

    assert metrics.node_count == 3
    assert metrics.edge_count == 3
    assert metrics.reciprocity > 0.0
    assert metrics.max_pair_share == 10.0 / 11.0
    assert metrics.collusion_index > 0.5
