from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from icframe.domain.base import Scalar
from icframe.domain.incentive_spec import Operation, OutcomeVector

from .types import RuntimeEvent, WorldSnapshot


@dataclass(frozen=True, slots=True)
class ResolvedStatePatch:
    target: str
    field: tuple[str, ...]
    operation: Operation
    value: Scalar


@dataclass(slots=True)
class HookResult:
    state_patches: list[ResolvedStatePatch] = field(default_factory=list)
    outcomes_by_agent: dict[str, OutcomeVector] = field(default_factory=dict)
    global_outcome: OutcomeVector = field(default_factory=dict)
    diagnostics: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class InitContext:
    snapshot: WorldSnapshot
    hook_config: dict[str, Scalar]
    rng: random.Random


@dataclass(frozen=True, slots=True)
class StepContext:
    snapshot: WorldSnapshot
    hook_config: dict[str, Scalar]
    rng: random.Random


@dataclass(frozen=True, slots=True)
class CommitContext:
    before: WorldSnapshot
    after: WorldSnapshot
    events: tuple[RuntimeEvent, ...]
    hook_config: dict[str, Scalar]
    rng: random.Random


class DomainHooks(Protocol):
    """Deterministic lifecycle extension; hooks never select or validate actions."""

    def initialize(self, context: InitContext) -> HookResult: ...

    def before_step(self, context: StepContext) -> HookResult: ...

    def after_commit(self, context: CommitContext) -> HookResult: ...

    def is_terminal(self, context: StepContext) -> bool: ...


class NoopHooks:
    def initialize(self, context: InitContext) -> HookResult:
        del context
        return HookResult()

    def before_step(self, context: StepContext) -> HookResult:
        del context
        return HookResult()

    def after_commit(self, context: CommitContext) -> HookResult:
        del context
        return HookResult()

    def is_terminal(self, context: StepContext) -> bool:
        del context
        return False
