from __future__ import annotations

import networkx as nx

from icframe.domain.evaluation import GraphMetrics
from icframe.domain.state import InteractionEdge, InteractionGraph, SimulationTrace
from icframe.ports.analyzer import AnalyzerPort


class NetworkXAnalyzer(AnalyzerPort):
    def project(self, trace: SimulationTrace) -> InteractionGraph:
        graph = nx.DiGraph()
        for agent in trace.final_snapshot.agents:
            graph.add_node(agent.name)

        for event in trace.events:
            if event.target is None or event.actor == event.target:
                continue
            weight = max(abs(event.amount) + abs(event.reward), 1.0)
            if graph.has_edge(event.actor, event.target):
                graph[event.actor][event.target]["weight"] += weight
                graph[event.actor][event.target]["event_count"] += 1
            else:
                graph.add_edge(event.actor, event.target, weight=weight, event_count=1)

        edges = [
            InteractionEdge(
                source=source,
                target=target,
                weight=float(data.get("weight", 0.0)),
                event_count=int(data.get("event_count", 0)),
            )
            for source, target, data in graph.edges(data=True)
        ]
        return InteractionGraph(nodes=tuple(sorted(graph.nodes())), edges=edges)

    def summarize(self, graph: InteractionGraph) -> GraphMetrics:
        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from(graph.nodes)
        for edge in graph.edges:
            nx_graph.add_edge(
                edge.source,
                edge.target,
                weight=edge.weight,
                event_count=edge.event_count,
            )

        total_weight = sum(data["weight"] for _, _, data in nx_graph.edges(data=True))
        if total_weight == 0:
            return GraphMetrics(node_count=nx_graph.number_of_nodes(), edge_count=0)

        reciprocity = nx.reciprocity(nx_graph)
        if reciprocity is None:
            reciprocity = 0.0

        pair_weights: list[float] = []
        for source, target, data in nx_graph.edges(data=True):
            if source >= target:
                continue
            reverse_weight = 0.0
            if nx_graph.has_edge(target, source):
                reverse_weight = float(nx_graph[target][source]["weight"])
            pair_weights.append(float(data["weight"]) + reverse_weight)

        max_pair_share = max((weight / total_weight for weight in pair_weights), default=0.0)
        collusion_index = float(reciprocity) * max_pair_share
        return GraphMetrics(
            node_count=nx_graph.number_of_nodes(),
            edge_count=nx_graph.number_of_edges(),
            reciprocity=float(reciprocity),
            max_pair_share=max_pair_share,
            collusion_index=collusion_index,
        )
