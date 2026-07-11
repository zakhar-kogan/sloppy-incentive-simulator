from __future__ import annotations

from typing import Any, Protocol

from icframe.domain.run import Checkpoint, RunSummary

from .types import Observation, PolicyDecision, RuntimeEvent


class RunObserver(Protocol):
    def cancelled(self) -> bool: ...

    def start(self, context: dict[str, Any]) -> None: ...

    def observation(self, value: Observation) -> None: ...

    def decision(self, value: PolicyDecision) -> None: ...

    def event(self, value: RuntimeEvent) -> None: ...

    def checkpoint(self, value: Checkpoint) -> None: ...

    def finish(self, value: RunSummary) -> None: ...


class NoopObserver:
    def cancelled(self) -> bool:
        return False

    def start(self, context: dict[str, Any]) -> None:
        del context

    def observation(self, value: Observation) -> None:
        del value

    def decision(self, value: PolicyDecision) -> None:
        del value

    def event(self, value: RuntimeEvent) -> None:
        del value

    def checkpoint(self, value: Checkpoint) -> None:
        del value

    def finish(self, value: RunSummary) -> None:
        del value
