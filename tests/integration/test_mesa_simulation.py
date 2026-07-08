from __future__ import annotations

import json
from pathlib import Path

from icframe.analytics.networkx import NetworkXAnalyzer
from icframe.pipelines import load_scenario, run_simulation, score_trace
from icframe.sim.mesa import MesaSimulator
from icframe.solvers.clingo import ClingoSolver


def test_mesa_simulation_is_deterministic_for_fixed_seed() -> None:
    scenario = load_scenario("examples/microbenches/public_goods.json")
    solver = ClingoSolver()
    simulator = MesaSimulator()
    analyzer = NetworkXAnalyzer()

    first_trace, _ = run_simulation(scenario, solver, simulator, analyzer, seed=7)
    second_trace, _ = run_simulation(scenario, solver, simulator, analyzer, seed=7)

    assert first_trace.model_dump(mode="json") == second_trace.model_dump(mode="json")


def test_mesa_simulation_matches_golden_trace_summary() -> None:
    scenario = load_scenario("examples/microbenches/public_goods.json")
    solver = ClingoSolver()
    simulator = MesaSimulator()
    analyzer = NetworkXAnalyzer()
    golden = json.loads(Path("tests/golden/public_goods_trace_summary.json").read_text())

    trace, _ = run_simulation(scenario, solver, simulator, analyzer, seed=7)
    evaluation = score_trace(scenario, trace, analyzer)

    final_balances = {agent.name: round(agent.balance, 6) for agent in trace.final_snapshot.agents}
    event_kinds = [event.kind.value for event in trace.events]

    assert final_balances == golden["final_balances"]
    assert event_kinds == golden["event_kinds"]
    assert round(evaluation.visible_score, 6) == golden["visible_score"]
