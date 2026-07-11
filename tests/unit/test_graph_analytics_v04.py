from __future__ import annotations

from icframe.analytics import analyze_interactions


def test_optional_graph_projection_uses_retained_events() -> None:
    summary = analyze_interactions(
        [
            {"actor_id": "a_000", "target_id": "b_000"},
            {"actor_id": "a_000", "target_id": "b_000"},
            {"actor_id": "b_000", "target_id": "a_000"},
            {"actor_id": "a_000", "target_id": None},
        ]
    )
    assert (summary.node_count, summary.edge_count, summary.reciprocity) == (2, 2, 1.0)
    assert summary.edges[0].count == 2
