from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from icframe import __version__
from icframe.domain.evaluation import EvaluationResult
from icframe.domain.mutations import FloatMutation, OptimizationResult, SearchSpace
from icframe.domain.norms import LawEvaluation
from icframe.domain.provenance import RunProvenance
from icframe.domain.reporting import AgentSeriesPoint, ExperimentSummary, StepSummary
from icframe.domain.scenario import Scenario
from icframe.domain.state import SimulationTrace
from icframe.pipelines.evaluate import evaluate_trace


def default_search_space() -> SearchSpace:
    return SearchSpace(
        float_params=[
            FloatMutation(name="contribution_bonus", low=0.0, high=4.0),
            FloatMutation(name="withhold_penalty", low=0.0, high=4.0),
            FloatMutation(name="tamper_penalty", low=1.0, high=10.0),
        ]
    )


def apply_best_params(scenario: Scenario, result: OptimizationResult) -> Scenario:
    mutated = scenario.model_copy(deep=True)
    for name, value in result.best_params.items():
        if hasattr(mutated.incentives, name):
            setattr(mutated.incentives, name, value)
        elif hasattr(mutated.simulation, name):
            setattr(mutated.simulation, name, value)
    return mutated


def build_experiment_summary(
    run_id: str,
    trace: SimulationTrace,
    evaluation: EvaluationResult,
    optimization: OptimizationResult | None = None,
) -> ExperimentSummary:
    event_counts = Counter(event.kind.value for event in trace.events)
    event_counts_by_step: dict[int, Counter[str]] = defaultdict(Counter)
    for event in trace.events:
        event_counts_by_step[event.step][event.kind.value] += 1

    agent_series = [
        AgentSeriesPoint(
            step=snapshot.step,
            name=agent.name,
            balance=agent.balance,
            payoff=agent.payoff,
            contributions=agent.contributions,
            withholds=agent.withholds,
            sent_messages=agent.sent_messages,
            received_messages=agent.received_messages,
            violations=agent.violations,
            last_action=agent.last_action,
        )
        for snapshot in trace.snapshots
        for agent in snapshot.agents
    ]
    step_summaries = [
        StepSummary(
            step=snapshot.step,
            public_pool=snapshot.public_pool,
            total_balance=sum(agent.balance for agent in snapshot.agents),
            total_payoff=sum(agent.payoff for agent in snapshot.agents),
            event_counts=dict(sorted(event_counts_by_step[snapshot.step].items())),
        )
        for snapshot in trace.snapshots
    ]

    graph_edges = trace.graph.edges if trace.graph is not None else []
    return ExperimentSummary(
        run_id=run_id,
        scenario_name=trace.scenario_name,
        seed=trace.seed,
        visible_score=evaluation.visible_score,
        trusted_score=evaluation.trusted_score,
        score_gap=evaluation.visible_score - evaluation.trusted_score,
        metrics=evaluation.metrics,
        diagnostics=evaluation.diagnostics,
        best_params=optimization.best_params if optimization else {},
        event_counts=dict(sorted(event_counts.items())),
        agent_outcomes=trace.final_snapshot.agents,
        agent_series=agent_series,
        step_summaries=step_summaries,
        graph_edges=graph_edges,
    )


def persist_run(
    output_dir: str | Path,
    scenario: Scenario,
    trace: SimulationTrace,
    evaluation: EvaluationResult,
    laws: LawEvaluation,
    optimization: OptimizationResult | None = None,
) -> RunProvenance:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "scenario.json").write_text(scenario.model_dump_json(indent=2))
    (output_path / "laws.json").write_text(laws.model_dump_json(indent=2))
    (output_path / "trace.json").write_text(trace.model_dump_json(indent=2))
    (output_path / "evaluation.json").write_text(evaluation.model_dump_json(indent=2))
    if optimization is not None:
        (output_path / "optimization.json").write_text(optimization.model_dump_json(indent=2))

    run_id = f"{scenario.name}-{trace.seed}"
    artifact_paths = [
        "evaluation.json",
        "laws.json",
        "provenance.json",
        "scenario.json",
        "summary.json",
        "trace.json",
    ]
    if optimization is not None:
        artifact_paths.append("optimization.json")

    provenance = RunProvenance(
        run_id=run_id,
        scenario_name=scenario.name,
        scenario_hash=hashlib.sha256(scenario.canonical_json().encode()).hexdigest(),
        seed=trace.seed,
        created_at=datetime.now(UTC),
        package_version=__version__,
        evaluation=evaluation,
        best_params=optimization.best_params if optimization else {},
        study_name=optimization.study_name if optimization else None,
        artifact_paths=sorted(artifact_paths),
    )
    summary = build_experiment_summary(run_id, trace, evaluation, optimization)
    (output_path / "summary.json").write_text(summary.model_dump_json(indent=2))
    (output_path / "provenance.json").write_text(provenance.model_dump_json(indent=2))
    return provenance


def score_trace(scenario: Scenario, trace: SimulationTrace, analyzer) -> EvaluationResult:
    if trace.graph is None:
        raise ValueError("trace is missing graph projection")
    graph_metrics = analyzer.summarize(trace.graph)
    return evaluate_trace(scenario, trace, graph_metrics)
