from __future__ import annotations

from typing import Protocol

from icframe.domain.norms import LawEvaluation
from icframe.domain.scenario import Scenario


class SolverPort(Protocol):
    def solve(self, scenario: Scenario) -> LawEvaluation:
        """Materialize allowed, forbidden, and violating actions for a scenario."""
