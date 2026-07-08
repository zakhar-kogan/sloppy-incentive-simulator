from __future__ import annotations

from typing import Protocol

from icframe.domain.evaluation import GraphMetrics
from icframe.domain.state import InteractionGraph, SimulationTrace


class AnalyzerPort(Protocol):
    def project(self, trace: SimulationTrace) -> InteractionGraph:
        """Project a simulation trace into a graph-shaped interaction summary."""

    def summarize(self, graph: InteractionGraph) -> GraphMetrics:
        """Compute graph-derived metrics without exposing backend graph types."""
