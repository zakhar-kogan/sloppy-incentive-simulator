from __future__ import annotations

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
from icframe.reports import write_html_report
from icframe.sim.mesa import MesaSimulator
from icframe.solvers.clingo import ClingoSolver


def test_html_report_is_generated_with_inline_charts(tmp_path) -> None:
    scenario = load_scenario("examples/microbenches/public_goods.json")
    solver = ClingoSolver()
    simulator = MesaSimulator()
    analyzer = NetworkXAnalyzer()
    optimizer = OptunaOptimizer(
        solver=solver,
        simulator=simulator,
        analyzer=analyzer,
        seed=7,
    )
    optimization = optimizer.optimize(scenario, default_search_space(), trials=2)
    tuned = apply_best_params(scenario, optimization)
    trace, laws = run_simulation(tuned, solver, simulator, analyzer, seed=7)
    evaluation = score_trace(tuned, trace, analyzer)
    persist_run(tmp_path, tuned, trace, evaluation, laws, optimization)

    report_path = write_html_report(tmp_path)
    html = report_path.read_text()

    assert report_path.name == "report.html"
    assert "ICFRAME experiment report" in html
    assert tuned.name in html
    assert "Event counts" in html
    assert "Balance trajectory" in html
    assert "Interaction graph" in html
    assert "<svg" in html
