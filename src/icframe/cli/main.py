from __future__ import annotations

import argparse
from pathlib import Path

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="icframe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate_parser = subparsers.add_parser(
        "simulate",
        help="run one scenario and persist artifacts",
    )
    simulate_parser.add_argument("scenario", type=Path)
    simulate_parser.add_argument("--seed", type=int, default=None)
    simulate_parser.add_argument("--output-dir", type=Path, required=True)

    optimize_parser = subparsers.add_parser(
        "optimize",
        help="search incentive parameters for one scenario",
    )
    optimize_parser.add_argument("scenario", type=Path)
    optimize_parser.add_argument("--trials", type=int, default=10)
    optimize_parser.add_argument("--seed", type=int, default=0)
    optimize_parser.add_argument("--output-dir", type=Path, required=True)

    report_parser = subparsers.add_parser(
        "report",
        help="render an HTML report from persisted artifacts",
    )
    report_parser.add_argument("artifact_dir", type=Path)
    report_parser.add_argument("--output", type=Path, default=None)
    return parser


def _compose_runtime() -> tuple[ClingoSolver, MesaSimulator, NetworkXAnalyzer]:
    return ClingoSolver(), MesaSimulator(), NetworkXAnalyzer()


def run_simulate(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    solver, simulator, analyzer = _compose_runtime()
    trace, laws = run_simulation(scenario, solver, simulator, analyzer, seed=args.seed)
    evaluation = score_trace(scenario, trace, analyzer)
    provenance = persist_run(args.output_dir, scenario, trace, evaluation, laws)
    print(f"persisted {provenance.run_id} to {args.output_dir}")
    print(
        f"visible_score={evaluation.visible_score:.3f} ",
        f"trusted_score={evaluation.trusted_score:.3f}",
        sep="",
    )
    return 0


def run_optimize(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    solver, simulator, analyzer = _compose_runtime()
    optimizer = OptunaOptimizer(
        solver=solver,
        simulator=simulator,
        analyzer=analyzer,
        seed=args.seed,
    )
    optimization = optimizer.optimize(
        scenario,
        default_search_space(),
        trials=args.trials,
    )
    tuned = apply_best_params(scenario, optimization)
    trace, laws = run_simulation(tuned, solver, simulator, analyzer, seed=args.seed)
    evaluation = score_trace(tuned, trace, analyzer)
    provenance = persist_run(
        args.output_dir,
        tuned,
        trace,
        evaluation,
        laws,
        optimization,
    )
    print(f"persisted {provenance.run_id} to {args.output_dir}")
    print(f"best_params={optimization.best_params}")
    print(f"best_trusted_score={optimization.best_value:.3f}")
    return 0


def run_report(args: argparse.Namespace) -> int:
    report_path = write_html_report(args.artifact_dir, args.output)
    print(f"wrote HTML report to {report_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "simulate":
        return run_simulate(args)
    if args.command == "report":
        return run_report(args)
    return run_optimize(args)


if __name__ == "__main__":
    raise SystemExit(main())
