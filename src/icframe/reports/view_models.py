from __future__ import annotations

from typing import Literal

from pydantic import Field

from icframe.domain.base import ICFrameModel, Scalar
from icframe.domain.run import RunSummary, StudySummary


class MetricView(ICFrameModel):
    id: str
    label: str
    value: float


class CheckpointView(ICFrameModel):
    step: int
    values: dict[str, float]


class AgentView(ICFrameModel):
    id: str
    archetype: str
    role: str
    state: str
    policy: str
    resources: dict[str, float]


class ConstraintView(ICFrameModel):
    metric: str
    value: float
    threshold: float
    operator: str
    passed: bool


class TrialView(ICFrameModel):
    number: int
    parameters: dict[str, Scalar]
    objectives: dict[str, float]
    feasible: bool
    state: str


class RunViewModel(ICFrameModel):
    kind: Literal["run"] = "run"
    id: str
    title: str
    status: str
    subtitle: str
    feasible: bool
    retention: str
    progress: float
    facts: dict[str, str]
    objectives: list[MetricView]
    metrics: list[MetricView]
    checkpoints: list[CheckpointView]
    actions: dict[str, int]
    tags: dict[str, int]
    agents: list[AgentView]
    constraints: list[ConstraintView]
    diagnostics: list[str] = Field(default_factory=list)


class StudyViewModel(ICFrameModel):
    kind: Literal["study"] = "study"
    id: str
    title: str
    status: str
    subtitle: str
    facts: dict[str, str]
    objectives: list[str]
    trials: list[TrialView]
    best_trial: int | None
    pareto_trials: list[int]
    retained_run_ids: list[str]


def run_view_model(summary: RunSummary) -> RunViewModel:
    diagnostics = []
    if summary.error:
        diagnostics.append(summary.error)
    if not summary.replayable:
        diagnostics.append(summary.replay_reason or "This run is not replayable.")
    if summary.llm_calls:
        diagnostics.append(
            f"{summary.llm_calls} LLM calls, estimated ${summary.estimated_llm_cost_usd:.4f}."
        )
    return RunViewModel(
        id=summary.run_id,
        title=summary.spec_name,
        status=summary.status.value,
        subtitle=f"{summary.pack_id} / seed {summary.seed}",
        feasible=summary.feasible,
        retention=summary.retention.value,
        progress=(
            summary.steps_completed / summary.steps_planned if summary.steps_planned else 0.0
        ),
        facts={
            "Steps": f"{summary.steps_completed:,} / {summary.steps_planned:,}",
            "Events": f"{summary.event_count:,}",
            "Duration": f"{summary.duration_seconds:.3f}s",
            "Replay": "ready" if summary.replayable else "unavailable",
        },
        objectives=[
            MetricView(id=name, label=_label(name), value=value)
            for name, value in sorted(summary.objectives.items())
        ],
        metrics=[
            MetricView(id=name, label=_label(name), value=value)
            for name, value in sorted(summary.metrics.items())
        ],
        checkpoints=[
            CheckpointView(step=item.step, values=dict(item.metrics))
            for item in summary.checkpoints
        ],
        actions=dict(summary.action_counts),
        tags=dict(summary.tag_counts),
        agents=[
            AgentView(
                id=item.id,
                archetype=item.archetype,
                role=item.role,
                state=item.state,
                policy=item.policy,
                resources=dict(item.resources),
            )
            for item in summary.agents
        ],
        constraints=[
            ConstraintView.model_validate(item.model_dump()) for item in summary.constraints
        ],
        diagnostics=diagnostics,
    )


def study_view_model(summary: StudySummary) -> StudyViewModel:
    return StudyViewModel(
        id=summary.study_id,
        title=f"{summary.pack_id} study",
        status=summary.status.value,
        subtitle=f"{summary.mode.value} / {summary.trial_count:,} trials",
        facts={
            "Trials": f"{summary.trial_count:,}",
            "Seeds": ", ".join(str(seed) for seed in summary.seeds),
            "Parameters": f"{len(summary.parameters):,}",
            "Duration": f"{summary.duration_seconds:.3f}s",
        },
        objectives=list(summary.objectives),
        trials=[
            TrialView(
                number=item.number,
                parameters=dict(item.parameters),
                objectives=dict(item.objective_values),
                feasible=item.feasible,
                state=item.state,
            )
            for item in summary.trials
        ],
        best_trial=summary.best_trial,
        pareto_trials=list(summary.pareto_trials),
        retained_run_ids=list(summary.retained_run_ids),
    )


def _label(value: str) -> str:
    return value.replace("_", " ").strip().title()
