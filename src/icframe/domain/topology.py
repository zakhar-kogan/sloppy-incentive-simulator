from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from .base import ICFrameModel


class CommunicationMode(StrEnum):
    ISOLATED = "isolated"
    LOCAL_K = "local_k"
    BROADCAST = "broadcast"
    COALITION = "coalition"


class CommunicationEdge(ICFrameModel):
    source: str
    target: str
    weight: float = Field(default=1.0, ge=0.0)


class TopologyConfig(ICFrameModel):
    mode: CommunicationMode = CommunicationMode.ISOLATED
    local_degree: int = Field(default=1, ge=0)
    coalition_members: tuple[str, ...] = ()
    edges: list[CommunicationEdge] = Field(default_factory=list)

    def materialize_edges(self, agent_names: tuple[str, ...]) -> list[CommunicationEdge]:
        if self.edges:
            return list(self.edges)

        if self.mode is CommunicationMode.ISOLATED:
            return []

        if self.mode is CommunicationMode.BROADCAST:
            return [
                CommunicationEdge(source=source, target=target)
                for source in agent_names
                for target in agent_names
                if source != target
            ]

        if self.mode is CommunicationMode.COALITION:
            members = set(self.coalition_members)
            return [
                CommunicationEdge(source=source, target=target)
                for source in agent_names
                for target in agent_names
                if source != target and source in members and target in members
            ]

        if not agent_names or self.local_degree == 0:
            return []

        edges: list[CommunicationEdge] = []
        degree = min(self.local_degree, max(len(agent_names) - 1, 0))
        for index, source in enumerate(agent_names):
            for offset in range(1, degree + 1):
                target = agent_names[(index + offset) % len(agent_names)]
                if source != target:
                    edges.append(CommunicationEdge(source=source, target=target))
        return edges
