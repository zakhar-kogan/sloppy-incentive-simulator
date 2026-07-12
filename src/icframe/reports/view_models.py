from __future__ import annotations

from typing import Literal

from pydantic import Field

from icframe.domain.base import ICFrameModel, Scalar
from icframe.domain.incentive_spec import (
    DomainPackManifest,
    IncentiveSpec,
    MetricCategory,
    MetricFormat,
    MetricType,
    PolicyKind,
    ReportMetric,
)
from icframe.domain.run import LLMUsageSummary, RunSummary, StudySummary


class MetricView(ICFrameModel):
    id: str
    label: str
    value: float
    description: str = ""
    unit: str | None = None
    format: str = MetricFormat.NUMBER.value
    desired_direction: str | None = None
    category: str = MetricCategory.OUTCOME.value
    formula: str | None = None
    cumulative: bool = False


class ChartGroupView(ICFrameModel):
    id: str
    label: str
    metrics: list[str]


class CheckpointView(ICFrameModel):
    step: int
    values: dict[str, float]
    action_counts: dict[str, int] = Field(default_factory=dict)


class AgentView(ICFrameModel):
    id: str
    archetype: str
    role: str
    state: str
    policy: str
    resources: dict[str, float]
    action_counts: dict[str, int] = Field(default_factory=dict)
    reward: float = 0.0
    failed_decisions: int = 0
    violations: int = 0
    detections: int = 0
    enforcement: int = 0


class ConstraintView(ICFrameModel):
    metric: str
    label: str
    value: float
    threshold: float
    operator: str
    passed: bool
    format: str = MetricFormat.NUMBER.value


class FindingView(ICFrameModel):
    kind: str
    text: str
    evidence: list[str] = Field(default_factory=list)


class MechanicsTransitionView(ICFrameModel):
    id: str
    label: str
    action: str
    from_state: str
    to_state: str
    tags: list[str]
    effects: list[str]
    enforcement: list[str]
    frequency: int = 0


class MechanicsView(ICFrameModel):
    states: list[str] = Field(default_factory=list)
    transitions: list[MechanicsTransitionView] = Field(default_factory=list)


class TrialView(ICFrameModel):
    number: int
    parameters: dict[str, Scalar]
    objectives: dict[str, float]
    feasible: bool
    state: str
    winner: bool = False
    frontier: bool = False


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
    findings: list[FindingView]
    objectives: list[MetricView]
    metrics: list[MetricView]
    chart_groups: list[ChartGroupView]
    checkpoints: list[CheckpointView]
    actions: dict[str, int]
    transitions: dict[str, int]
    tags: dict[str, int]
    agents: list[AgentView]
    constraints: list[ConstraintView]
    mechanics: MechanicsView
    llm: LLMUsageSummary
    has_llm: bool = False
    diagnostics: list[str] = Field(default_factory=list)


class StudyViewModel(ICFrameModel):
    kind: Literal["study"] = "study"
    id: str
    title: str
    status: str
    subtitle: str
    facts: dict[str, str]
    findings: list[FindingView]
    objectives: list[str]
    objective_presentations: list[MetricView]
    trials: list[TrialView]
    best_trial: int | None
    pareto_trials: list[int]
    retained_run_ids: list[str]


def run_view_model(
    summary: RunSummary,
    manifest: DomainPackManifest | None = None,
    spec: IncentiveSpec | None = None,
) -> RunViewModel:
    diagnostics = []
    if summary.error:
        diagnostics.append(summary.error)
    if not summary.replayable:
        diagnostics.append(summary.replay_reason or "This run is not replayable.")
    llm = _llm_usage(summary)
    if llm.attempted:
        cost = (
            f"${llm.estimated_cost_usd:.4f}"
            if llm.estimated_cost_usd is not None
            else "cost unavailable"
        )
        diagnostics.append(f"{llm.attempted} LLM attempts, {cost}.")
    metrics = [
        _metric_view(name, value, manifest, spec)
        for name, value in sorted(
            summary.metrics.items(),
            key=lambda item: _metric_sort_key(item[0], manifest),
        )
    ]
    objective_metrics = []
    for name, value in summary.objectives.items():
        metric_name = (
            spec.evaluation.objectives[name].metric
            if spec is not None and name in spec.evaluation.objectives
            else name
        )
        objective_metrics.append(_metric_view(metric_name, value, manifest, spec, id=name))
    transition_labels = _transition_labels(spec)
    constraints = [
        ConstraintView(
            metric=item.metric,
            label=_presentation(item.metric, manifest).label,
            value=item.value,
            threshold=item.threshold,
            operator=item.operator,
            passed=item.passed,
            format=_presentation(item.metric, manifest).format.value,
        )
        for item in summary.constraints
    ]
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
        findings=_run_findings(summary, metrics, constraints, transition_labels),
        objectives=objective_metrics,
        metrics=metrics,
        chart_groups=_chart_groups(metrics, manifest),
        checkpoints=[
            CheckpointView(
                step=item.step,
                values=dict(item.metrics),
                action_counts=dict(item.action_counts),
            )
            for item in summary.checkpoints
        ],
        actions=dict(summary.action_counts),
        transitions=dict(summary.transition_counts),
        tags=dict(summary.tag_counts),
        agents=[
            AgentView(
                id=item.id,
                archetype=item.archetype,
                role=item.role,
                state=item.state,
                policy=item.policy,
                resources=dict(item.resources),
                action_counts=dict(item.statistics.action_counts),
                reward=item.statistics.reward,
                failed_decisions=item.statistics.failed_decisions,
                violations=item.statistics.violations,
                detections=item.statistics.detections,
                enforcement=item.statistics.enforcement,
            )
            for item in summary.agents
        ],
        constraints=constraints,
        mechanics=_mechanics(spec, summary.transition_counts),
        llm=llm,
        has_llm=bool(
            llm.attempted
            or (
                spec is not None
                and any(item.policy is PolicyKind.LLM for item in spec.archetypes.values())
            )
        ),
        diagnostics=diagnostics,
    )


def study_view_model(
    summary: StudySummary,
    manifest: DomainPackManifest | None = None,
    spec: IncentiveSpec | None = None,
) -> StudyViewModel:
    objective_presentations = [
        _metric_view(
            spec.evaluation.objectives[name].metric
            if spec is not None and name in spec.evaluation.objectives
            else name,
            0.0,
            manifest,
            spec,
            id=name,
        )
        for name in summary.objectives
    ]
    feasible_count = sum(item.feasible for item in summary.trials)
    findings = [
        FindingView(
            kind="feasibility",
            text=f"{feasible_count} of {summary.trial_count} trials were feasible.",
            evidence=["trials"],
        )
    ]
    if summary.best_trial is not None:
        findings.append(
            FindingView(
                kind="winner",
                text=f"Trial {summary.best_trial} is the selected winner.",
                evidence=[f"trial:{summary.best_trial}"],
            )
        )
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
        findings=findings,
        objectives=list(summary.objectives),
        objective_presentations=objective_presentations,
        trials=[
            TrialView(
                number=item.number,
                parameters=dict(item.parameters),
                objectives=dict(item.objective_values),
                feasible=item.feasible,
                state=item.state,
                winner=item.number == summary.best_trial,
                frontier=item.number in summary.pareto_trials,
            )
            for item in summary.trials
        ],
        best_trial=summary.best_trial,
        pareto_trials=list(summary.pareto_trials),
        retained_run_ids=list(summary.retained_run_ids),
    )


def _presentation(name: str, manifest: DomainPackManifest | None) -> ReportMetric:
    if manifest is not None and name in manifest.report.metrics:
        return manifest.report.metrics[name]
    return ReportMetric(label=_label(name))


def _metric_view(
    name: str,
    value: float,
    manifest: DomainPackManifest | None,
    spec: IncentiveSpec | None,
    *,
    id: str | None = None,
) -> MetricView:
    presentation = _presentation(name, manifest)
    metric = spec.metrics.get(name) if spec is not None else None
    cumulative = bool(
        metric and metric.type in {MetricType.SUM, MetricType.EVENT_COUNT, MetricType.WEIGHTED_SUM}
    )
    return MetricView(
        id=id or name,
        label=presentation.label,
        value=value,
        description=presentation.description,
        unit=presentation.unit,
        format=presentation.format.value,
        desired_direction=(
            presentation.desired_direction.value
            if presentation.desired_direction is not None
            else None
        ),
        category=presentation.category.value,
        formula=_formula(name, spec) if spec is not None and name in spec.metrics else None,
        cumulative=cumulative,
    )


def _metric_sort_key(name: str, manifest: DomainPackManifest | None) -> tuple[int, str]:
    presentation = _presentation(name, manifest)
    return (
        presentation.headline_order if presentation.headline_order is not None else 1_000_000,
        name,
    )


def _chart_groups(
    metrics: list[MetricView], manifest: DomainPackManifest | None
) -> list[ChartGroupView]:
    if manifest is not None and manifest.report.chart_groups:
        return [
            ChartGroupView.model_validate(item.model_dump())
            for item in manifest.report.chart_groups
        ]
    return [ChartGroupView(id="metrics", label="Metrics", metrics=[item.id for item in metrics])]


def _run_findings(
    summary: RunSummary,
    metrics: list[MetricView],
    constraints: list[ConstraintView],
    transition_labels: dict[str, str],
) -> list[FindingView]:
    findings = []
    if constraints:
        passed = sum(item.passed for item in constraints)
        findings.append(
            FindingView(
                kind="constraint",
                text=f"{passed} of {len(constraints)} trusted constraints passed.",
                evidence=[f"constraint:{item.metric}" for item in constraints],
            )
        )
    if metrics:
        primary = metrics[0]
        findings.append(
            FindingView(
                kind="outcome",
                text=f"{primary.label} finished at {_format_value(primary.value, primary.format)}.",
                evidence=[f"metric:{primary.id}"],
            )
        )
    if summary.transition_counts:
        transition_id, count = max(
            summary.transition_counts.items(), key=lambda item: (item[1], item[0])
        )
        findings.append(
            FindingView(
                kind="behavior",
                text=(
                    f"{transition_labels.get(transition_id, _label(transition_id))} "
                    f"was the most frequent action ({count:,})."
                ),
                evidence=[f"transition:{transition_id}"],
            )
        )
    if len(summary.checkpoints) >= 2:
        first, last = summary.checkpoints[0], summary.checkpoints[-1]
        for metric in metrics[:3]:
            start = first.metrics.get(metric.id)
            end = last.metrics.get(metric.id)
            if start is None or end is None or start == end:
                continue
            verb = "rose" if end > start else "fell"
            findings.append(
                FindingView(
                    kind="trend",
                    text=(
                        f"{metric.label} {verb} from {_format_value(start, metric.format)} "
                        f"to {_format_value(end, metric.format)}."
                    ),
                    evidence=[f"metric:{metric.id}"],
                )
            )
    enforcement = sum(item.statistics.enforcement for item in summary.agents)
    if enforcement:
        findings.append(
            FindingView(
                kind="enforcement",
                text=f"Enforcement was applied {enforcement:,} times.",
                evidence=["agents:enforcement"],
            )
        )
    return findings[:7]


def _mechanics(spec: IncentiveSpec | None, counts: dict[str, int]) -> MechanicsView:
    if spec is None:
        return MechanicsView()
    transitions = []
    for item in spec.transitions:
        effects = [
            f"{effect.scope.value}: "
            + ", ".join(f"{name} {value:+g}" for name, value in effect.values.items())
            for effect in item.effects
        ]
        enforcement = []
        if item.enforcement is not None:
            enforcement.append(f"audit {item.enforcement.audit_probability:.0%}")
            enforcement.extend(
                f"sanction {effect.scope.value}: "
                + ", ".join(f"{name} {value:+g}" for name, value in effect.values.items())
                for effect in item.enforcement.sanctions
            )
        transitions.append(
            MechanicsTransitionView(
                id=item.id,
                label=item.prompt.label
                if item.prompt and item.prompt.label
                else _label(item.action),
                action=item.action,
                from_state=item.from_state,
                to_state=item.to_state,
                tags=list(item.tags),
                effects=effects,
                enforcement=enforcement,
                frequency=counts.get(item.id, 0),
            )
        )
    return MechanicsView(states=list(spec.states.all), transitions=transitions)


def _formula(name: str, spec: IncentiveSpec, seen: set[str] | None = None) -> str:
    seen = set(seen or ())
    if name in seen:
        return name
    seen.add(name)
    metric = spec.metrics[name]
    if metric.type is MetricType.WEIGHTED_SUM:
        return " + ".join(
            f"{weight:g} x ({_formula(reference, spec, seen)})"
            for reference, weight in metric.terms.items()
        )
    if metric.type is MetricType.DIFFERENCE:
        left = _formula(metric.left or "", spec, seen)
        right = _formula(metric.right or "", spec, seen)
        return f"({left}) - ({right})"
    if metric.type is MetricType.RATIO:
        numerator = _formula(metric.numerator or "", spec, seen)
        denominator = _formula(metric.denominator or "", spec, seen)
        return f"({numerator}) / ({denominator})"
    if metric.type in {MetricType.EVENT_COUNT, MetricType.EVENT_RATE}:
        tags = ", ".join(metric.where_tags_include) or "all actions"
        return f"{metric.type.value}({tags})"
    return f"{metric.type.value}({metric.channel})"


def _transition_labels(spec: IncentiveSpec | None) -> dict[str, str]:
    if spec is None:
        return {}
    return {
        item.id: item.prompt.label if item.prompt and item.prompt.label else _label(item.action)
        for item in spec.transitions
    }


def _llm_usage(summary: RunSummary) -> LLMUsageSummary:
    usage = summary.llm_usage.model_copy(deep=True)
    if usage.attempted or not summary.llm_calls:
        return usage
    usage.attempted = summary.llm_calls
    usage.completed = summary.llm_calls
    usage.estimated_cost_usd = (
        summary.estimated_llm_cost_usd
        if summary.estimated_llm_cost_usd not in {None, 0.0}
        else None
    )
    return usage


def _format_value(value: float, format_name: str) -> str:
    if format_name == MetricFormat.PERCENT.value:
        return f"{value:.1%}"
    if format_name == MetricFormat.CURRENCY.value:
        return f"${value:,.2f}"
    if format_name == MetricFormat.INTEGER.value:
        return f"{value:,.0f}"
    if format_name == MetricFormat.DURATION.value:
        return f"{value:,.2f}s"
    return f"{value:,.4g}"


def _label(value: str) -> str:
    return value.replace("_", " ").strip().title()
