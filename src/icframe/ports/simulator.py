from __future__ import annotations

from typing import Protocol

from icframe.domain.norms import LawEvaluation
from icframe.domain.scenario import Scenario
from icframe.domain.state import SimulationTrace


class SimulatorPort(Protocol):
    def run(
        self,
        scenario: Scenario,
        laws: LawEvaluation,
        seed: int | None = None,
    ) -> SimulationTrace:
        """Run a scenario under a law evaluation and return a trace."""
