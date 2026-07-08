from __future__ import annotations

import json

from icframe.analytics.networkx import NetworkXAnalyzer
from icframe.optimize.optuna import OptunaOptimizer
from icframe.pipelines import (
    apply_best_params,
    default_search_space,
    load_scenario,
    persist_run,
    run_simulation,
    score_trace,
)
from icframe.sim.mesa import MesaSimulator
from icframe.solvers.clingo import ClingoSolver


def test_end_to_end_pipeline_persists_provenance_and_optimization(tmp_path) -> None:
    scenario = load_scenario("examples/microbenches/public_goods.json")
    solver = ClingoSolver()
    simulator = MesaSimulator()
    analyzer = NetworkXAnalyzer()

    trace, _laws = run_simulation(scenario, solver, simulator, analyzer, seed=7)
    evaluation = score_trace(scenario, trace, analyzer)
    optimizer = OptunaOptimizer(
        solver=solver,
        simulator=simulator,
        analyzer=analyzer,
        seed=7,
    )
    optimization = optimizer.optimize(scenario, default_search_space(), trials=3)
    tuned_scenario = apply_best_params(scenario, optimization)
    tuned_trace, tuned_laws = run_simulation(
        tuned_scenario,
        solver,
        simulator,
        analyzer,
        seed=7,
    )
    tuned_evaluation = score_trace(tuned_scenario, tuned_trace, analyzer)

    provenance = persist_run(
        tmp_path,
        tuned_scenario,
        tuned_trace,
        tuned_evaluation,
        tuned_laws,
        optimization,
    )
    optimization_payload = json.loads((tmp_path / "optimization.json").read_text())
    provenance_payload = json.loads((tmp_path / "provenance.json").read_text())
    summary_payload = json.loads((tmp_path / "summary.json").read_text())

    assert evaluation.trusted_score > 0
    assert len(optimization.trials) == 3
    assert set(optimization.best_params)
    assert optimization.best_value >= evaluation.trusted_score
    assert provenance.study_name == optimization.study_name
    assert optimization_payload["best_params"] == optimization.best_params
    assert summary_payload["run_id"] == provenance.run_id
    assert summary_payload["event_counts"]["contribute"] >= 1
    assert sorted(provenance_payload["artifact_paths"]) == sorted(
        [
            "evaluation.json",
            "laws.json",
            "optimization.json",
            "provenance.json",
            "scenario.json",
            "summary.json",
            "trace.json",
        ]
    )
