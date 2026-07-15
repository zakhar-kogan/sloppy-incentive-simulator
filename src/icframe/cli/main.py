from __future__ import annotations

import argparse
import json
from pathlib import Path

from icframe.catalog import Catalog
from icframe.core import list_domain_packs, load_domain_pack, run_experiment
from icframe.domain.base import Scalar
from icframe.domain.incentive_spec import RetentionProfile
from icframe.domain.run import LiveLLMBudget, PlannerKind, RunConfig, StudyConfig, StudyMode
from icframe.llm import LiteLLMClient, LLMClient
from icframe.orchestration import submit_run, submit_study
from icframe.planning import apply_study_preset
from icframe.profiles import apply_llm_profile, llm_client_for_profile, load_profiles
from icframe.replay import replay_run
from icframe.reports import write_html_report
from icframe.runtime_settings import load_runtime_llm_settings
from icframe.study import run_study
from icframe.ui.server import serve_ui


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="icframe")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("packs", help="list installed domain packs")

    run = commands.add_parser("run", help="run one v0.4 domain-pack experiment")
    run.add_argument("pack")
    run.add_argument("--seed", type=int)
    run.add_argument("--param", action="append", default=[], metavar="ID=VALUE")
    run.add_argument("--retention", choices=[item.value for item in RetentionProfile])
    run.add_argument("--sample-every-steps", type=int)
    _artifact_argument(run)
    _backend_arguments(run)
    _llm_arguments(run)

    study = commands.add_parser("study", help="run a planned single or Pareto study")
    study.add_argument("pack")
    study.add_argument("--mode", choices=[item.value for item in StudyMode])
    study.add_argument("--objective", action="append", default=[])
    study.add_argument("--parameter", action="append", default=[])
    study.add_argument("--trials", type=int, default=20)
    study.add_argument("--planner", choices=["matrix", "random"], default="random")
    study.add_argument("--preset", help="named study preset from the domain manifest")
    study.add_argument("--planner-seed", type=int, default=0)
    study.add_argument(
        "--matrix",
        action="append",
        default=[],
        metavar="ID=JSON_VALUES",
        help="matrix values such as audit_probability=[0,0.5,1]",
    )
    study.add_argument("--seeds", help="comma-separated integer seeds")
    study.add_argument("--workers", type=int)
    study.add_argument("--allow-live-llm", action="store_true")
    study.add_argument("--max-llm-calls", type=int)
    study.add_argument("--max-llm-cost-usd", type=float)
    _artifact_argument(study)
    _backend_arguments(study)
    _llm_arguments(study)

    replay = commands.add_parser("replay", help="replay a retained run")
    replay.add_argument("artifact_dir", type=Path)

    report = commands.add_parser("report", help="export a self-contained HTML report")
    report.add_argument("artifact", type=Path)
    report.add_argument("--output", type=Path)

    catalog = commands.add_parser("catalog", help="manage the rebuildable artifact index")
    catalog.add_argument("operation", choices=["rebuild"])
    _artifact_argument(catalog)

    ui = commands.add_parser("ui", help="launch the canonical local simulator UI")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    _artifact_argument(ui)

    worker = commands.add_parser("worker", help="run a versioned ICFRAME worker contract")
    worker_commands = worker.add_subparsers(dest="worker_command", required=True)
    execute_shard = worker_commands.add_parser("execute-shard", help="execute one study shard")
    execute_shard.add_argument("--request", type=Path, required=True)
    execute_shard.add_argument("--output", type=Path, required=True)
    return parser


def _artifact_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(".artifacts/icframe"),
    )


def _llm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-mode", choices=["none", "live"], default="none")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-profile")


def _backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execution-profile", default="local")


def run_pack(args: argparse.Namespace) -> int:
    pack = _load_pack(args.pack)
    if args.llm_profile:
        pack = apply_llm_profile(pack, load_profiles().llm_profile(args.llm_profile))
    config = RunConfig(
        seed=args.seed,
        parameters=_parameters(args.param),
        retention=(RetentionProfile(args.retention) if args.retention else None),
        sample_every_steps=args.sample_every_steps,
        artifact_root=args.artifact_root,
    )
    if args.execution_profile != "local":
        handle = submit_run(
            pack,
            config,
            backend_profile=args.execution_profile,
            llm_profile=args.llm_profile,
        )
        print(handle.model_dump_json(indent=2))
        return 0
    summary = run_experiment(
        pack,
        config,
        llm_client=_llm_client(args),
    )
    print(summary.model_dump_json(indent=2))
    return 0


def run_optimization(args: argparse.Namespace) -> int:
    pack = _load_pack(args.pack)
    if args.llm_profile:
        pack = apply_llm_profile(pack, load_profiles().llm_profile(args.llm_profile))
    preset = next(
        (item for item in pack.manifest.study.presets if item.id == args.preset),
        None,
    )
    if args.preset and preset is None:
        raise ValueError(f"unknown study preset: {args.preset}")
    if preset is not None:
        pack = apply_study_preset(pack, preset)
    mode_value = args.mode
    if mode_value is None:
        mode_value = (
            "single" if preset is not None and len(preset.objectives) == 1 else "pareto"
        ) if preset is not None else "single"
    mode = StudyMode(mode_value)
    objectives = args.objective or (list(preset.objectives) if preset else None) or (
        [pack.manifest.study.single_objective]
        if mode is StudyMode.SINGLE
        else list(pack.manifest.study.pareto_objectives)
    )
    parameters = args.parameter or (
        list(preset.parameters or preset.parameter_matrix) if preset else []
    ) or [item.id for item in pack.manifest.parameters if item.optimizable]
    seeds = (
        [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else list(preset.seeds if preset else pack.spec.experiment.seeds)
    )
    llm_client = _llm_client(args)
    if args.llm_mode == "live" and not args.allow_live_llm:
        raise ValueError("live LLM studies require --allow-live-llm and explicit budgets")
    budget = LiveLLMBudget(
        enabled=args.allow_live_llm,
        max_calls=args.max_llm_calls,
        max_cost_usd=args.max_llm_cost_usd,
    )
    config_values: dict[str, object] = {
        "mode": mode,
        "objectives": objectives,
        "parameters": parameters,
        "trials": preset.trials or args.trials if preset else args.trials,
        "seeds": seeds,
        "artifact_root": args.artifact_root,
        "live_llm": budget,
        "planner": PlannerKind(preset.planner if preset else args.planner),
        "planner_seed": args.planner_seed,
        "parameter_matrix": (
            dict(preset.parameter_matrix) if preset else _parameter_matrix(args.matrix)
        ),
    }
    if args.workers is not None:
        config_values["workers"] = args.workers
    config = StudyConfig(**config_values)
    if args.execution_profile != "local":
        handle = submit_study(
            pack,
            config,
            backend_profile=args.execution_profile,
            llm_profile=args.llm_profile,
        )
        print(handle.model_dump_json(indent=2))
        return 0
    summary = run_study(pack, config, llm_client=llm_client)
    print(summary.model_dump_json(indent=2))
    return 0


def run_packs() -> int:
    for manifest in list_domain_packs():
        print(f"{manifest.pack.id}\t{manifest.pack.title}\t{len(manifest.parameters)} parameters")
    return 0


def _load_pack(value: str):
    path = Path(value)
    if path.suffix.lower() == ".json" or (path.is_file() and path.name != "pack.toml"):
        raise ValueError(
            "legacy scenarios and standalone specs are unsupported; use an IncentiveSpec "
            "v0.4 domain pack containing pack.toml"
        )
    return load_domain_pack(value)


def _parameters(values: list[str]) -> dict[str, Scalar]:
    parameters: dict[str, Scalar] = {}
    for assignment in values:
        key, separator, raw = assignment.partition("=")
        if not separator or not key:
            raise ValueError(f"parameter override must use ID=VALUE: {assignment!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if not isinstance(value, str | int | float | bool):
            raise ValueError(f"parameter {key} must be a scalar")
        parameters[key] = value
    return parameters


def _parameter_matrix(values: list[str]) -> dict[str, list[Scalar]]:
    matrix: dict[str, list[Scalar]] = {}
    for assignment in values:
        key, separator, raw = assignment.partition("=")
        if not separator or not key:
            raise ValueError(f"matrix assignment must use ID=JSON_VALUES: {assignment!r}")
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError(f"matrix parameter {key} requires a non-empty JSON list")
        if not all(isinstance(item, str | int | float | bool) for item in parsed):
            raise ValueError(f"matrix parameter {key} values must be scalars")
        matrix[key] = parsed
    return matrix


def _llm_client(args: argparse.Namespace) -> LLMClient | None:
    if args.llm_profile:
        return llm_client_for_profile(load_profiles().llm_profile(args.llm_profile))
    if args.llm_mode == "live":
        return LiteLLMClient(
            load_runtime_llm_settings(
                base_url=args.llm_base_url,
                model=args.llm_model,
                api_key_source="shell environment or .env",
            )
        )
    return None


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "packs":
        return run_packs()
    if args.command == "run":
        return run_pack(args)
    if args.command == "study":
        return run_optimization(args)
    if args.command == "replay":
        summary = replay_run(args.artifact_dir)
        print(summary.model_dump_json(indent=2))
        return 0
    if args.command == "report":
        print(write_html_report(args.artifact, args.output))
        return 0
    if args.command == "catalog":
        print(json.dumps(Catalog(args.artifact_root).rebuild(), sort_keys=True))
        return 0
    if args.command == "worker":
        from icframe.orchestration.worker import execute_study_shard_file

        print(execute_study_shard_file(args.request, args.output))
        return 0
    serve_ui(host=args.host, port=args.port, artifact_root=args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
