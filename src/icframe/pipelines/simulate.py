from __future__ import annotations

from pathlib import Path

from icframe.domain.norms import LawEvaluation
from icframe.domain.scenario import Scenario
from icframe.domain.state import SimulationTrace
from icframe.ports.analyzer import AnalyzerPort
from icframe.ports.simulator import SimulatorPort
from icframe.ports.solver import SolverPort


def load_scenario(path: str | Path) -> Scenario:
    return Scenario.model_validate_json(Path(path).read_text())


def run_simulation(
    scenario: Scenario,
    solver: SolverPort,
    simulator: SimulatorPort,
    analyzer: AnalyzerPort,
    seed: int | None = None,
) -> tuple[SimulationTrace, LawEvaluation]:
    laws = solver.solve(scenario)
    trace = simulator.run(scenario, laws, seed=seed)
    trace.graph = analyzer.project(trace)
    return trace, laws
