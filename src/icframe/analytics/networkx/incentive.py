from __future__ import annotations

from collections import Counter

from icframe.domain.state import InteractionEdge, InteractionGraph
from icframe.runtime.incentive import SimulationTrace


def project_incentive_trace(trace: SimulationTrace) -> InteractionGraph:
    """Project v0.2 events into a lightweight actor-to-action interaction graph."""

    nodes = set(trace.final_agent_state)
    edge_counts: Counter[tuple[str, str]] = Counter()
    for event in trace.events:
        action_node = f"action:{event.action}"
        nodes.add(action_node)
        edge_counts[(event.actor_id, action_node)] += 1
    edges = [
        InteractionEdge(
            source=source,
            target=target,
            weight=float(count),
            event_count=count,
        )
        for (source, target), count in sorted(edge_counts.items())
    ]
    return InteractionGraph(nodes=tuple(sorted(nodes)), edges=edges)
