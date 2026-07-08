from __future__ import annotations

from typing import Protocol

from icframe.domain.mutations import OptimizationResult, SearchSpace
from icframe.domain.scenario import Scenario


class OptimizerPort(Protocol):
    def optimize(
        self,
        scenario: Scenario,
        search_space: SearchSpace,
        trials: int,
    ) -> OptimizationResult:
        """Search for better incentive parameters under a fixed trusted evaluator."""
