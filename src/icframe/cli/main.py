from __future__ import annotations

import argparse
import json
from pathlib import Path

from icframe.catalog import Catalog
from icframe.core import list_domain_packs, load_domain_pack, run_experiment
from icframe.domain.base import Scalar
from icframe.domain.incentive_spec import RetentionProfile
from icframe.domain.run import LiveLLMBudget, RunConfig, StudyConfig, StudyMode
from icframe.llm import LiteLLMClient, LLMClient
from icframe.replay import replay_run
from icframe.reports import write_html_report
from icframe.runtime_settings import load_runtime_llm_settings
from icframe.study import run_study


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
    _llm_arguments(run)

    study = commands.add_parser("study", help="run a single or Pareto Optuna study")
    study.add_argument("pack")
    study.add_argument("--mode", choices=[item.value for item in StudyMode], default="single")
    study.add_argument("--objective", action="append", default=[])
    study.add_argument("--parameter", action="append", default=[])
    study.add_argument("--trials", type=int, default=20)
    study.add_argument("--seeds", help="comma-separated integer seeds")
    study.add_argument("--workers", type=int)
    study.add_argument("--allow-live-llm", action="store_true")
    study.add_argument("--max-llm-calls", type=int)
    study.add_argument("--max-llm-cost-usd", type=float)
    _artifact_argument(study)
    _llm_arguments(study)

    replay = commands.add_parser("replay", help="replay a retained run")
    replay.add_argument("artifact_dir", type=Path)

    report = commands.add_parser("report", help="export a self-contained HTML report")
    report.add_argument("artifact", type=Path)
    report.add_argument("--output", type=Path)

    catalog = commands.add_parser("catalog", help="manage the rebuildable artifact index")
    catalog.add_argument("operation", choices=["rebuild"])
    _artifact_argument(catalog)

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


def run_pack(args: argparse.Namespace) -> int:
    pack = _load_pack(args.pack)
    summary = run_experiment(
        pack,
        RunConfig(
            seed=args.seed,
            parameters=_parameters(args.param),
            retention=(RetentionProfile(args.retention) if args.retention else None),
            sample_every_steps=args.sample_every_steps,
            artifact_root=args.artifact_root,
        ),
        llm_client=_llm_client(args),
    )
    print(summary.model_dump_json(indent=2))
    return 0


def run_optimization(args: argparse.Namespace) -> int:
    pack = _load_pack(args.pack)
    mode = StudyMode(args.mode)
    objectives = args.objective or (
        [pack.manifest.study.single_objective]
        if mode is StudyMode.SINGLE
        else list(pack.manifest.study.pareto_objectives)
    )
    parameters = args.parameter or [
        item.id for item in pack.manifest.parameters if item.optimizable
    ]
    seeds = (
        [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else list(pack.spec.experiment.seeds)
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
        "trials": args.trials,
        "seeds": seeds,
        "artifact_root": args.artifact_root,
        "live_llm": budget,
    }
    if args.workers is not None:
        config_values["workers"] = args.workers
    summary = run_study(pack, StudyConfig(**config_values), llm_client=llm_client)
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


def _llm_client(args: argparse.Namespace) -> LLMClient | None:
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
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
