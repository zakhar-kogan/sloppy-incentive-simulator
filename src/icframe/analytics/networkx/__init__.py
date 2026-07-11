from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import Field

from icframe.domain.base import ICFrameModel


class InteractionEdge(ICFrameModel):
    source: str
    target: str
    count: int


class InteractionGraphSummary(ICFrameModel):
    node_count: int
    edge_count: int
    density: float
    reciprocity: float
    components: int
    edges: list[InteractionEdge] = Field(default_factory=list)


def analyze_interactions(
    source: str | Path | Iterable[dict[str, Any]],
) -> InteractionGraphSummary:
    """Project retained target-directed events into an optional NetworkX graph."""

    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - base installation
        raise RuntimeError("install icframe[analytics] to use graph analytics") from exc

    graph = nx.DiGraph()
    for event in _events(source):
        actor = event.get("actor_id")
        target = event.get("target_id")
        if not isinstance(actor, str) or not isinstance(target, str):
            continue
        count = int(graph.get_edge_data(actor, target, {}).get("count", 0)) + 1
        graph.add_edge(actor, target, count=count)
    reciprocity = float(nx.reciprocity(graph) or 0.0) if graph.number_of_edges() else 0.0
    components = nx.number_weakly_connected_components(graph) if graph.number_of_nodes() else 0
    return InteractionGraphSummary(
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        density=float(nx.density(graph)),
        reciprocity=reciprocity,
        components=components,
        edges=[
            InteractionEdge(source=source, target=target, count=int(data["count"]))
            for source, target, data in sorted(graph.edges(data=True))
        ],
    )


def _events(source: str | Path | Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    if not isinstance(source, str | Path):
        yield from source
        return
    path = Path(source)
    for line in path.read_text().splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload
