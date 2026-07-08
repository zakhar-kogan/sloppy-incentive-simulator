from __future__ import annotations

import json

import clingo
from pydantic import Field

from icframe.domain.base import ICFrameModel
from icframe.domain.incentive_spec import IncentiveSpec, NormStatus, Transition


class ConstraintProblem(ICFrameModel):
    subject: str
    code: str
    message: str


class ConstraintReport(ICFrameModel):
    ok: bool
    problems: list[ConstraintProblem] = Field(default_factory=list)


class ConstraintExplanation(ICFrameModel):
    constraint_id: str | None = None
    policy_decision_id: str | None = None
    event_id: str | None = None
    actor_id: str
    state: str
    action: str
    transition_id: str | None = None
    available: bool = False
    hard_blocked: bool = False
    blocked: bool = False
    norm_status: NormStatus = NormStatus.UNKNOWN
    reasons: list[str] = Field(default_factory=list)
    obligations: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    remediation_actions: list[str] = Field(default_factory=list)
    explanation_facts: list[str] = Field(default_factory=list)


def validate_constraints(spec: IncentiveSpec) -> ConstraintReport:
    facts = _facts_for_spec(spec)
    program = "\n".join(
        [
            *facts,
            (
                'problem(T,"hard_blocked_transition") :- '
                'transition(T,_,_,_), availability(T,"hard_blocked").'
            ),
            (
                'problem(T,"auditable_forbidden_missing_enforcement") :- '
                'transition(T,_,_,_), norm_status(T,"forbidden"), '
                'tag(T,"auditable"), not has_enforcement(T).'
            ),
            "#show problem/2.",
        ]
    )
    symbols = _solve(program)
    problems = [
        ConstraintProblem(
            subject=_string_arg(symbol, 0),
            code=_string_arg(symbol, 1),
            message=_problem_message(_string_arg(symbol, 1)),
        )
        for symbol in symbols
        if symbol.name == "problem"
    ]
    return ConstraintReport(ok=not problems, problems=problems)


def explain_transition_availability(
    spec: IncentiveSpec,
    actor_id: str,
    state: str,
    action: str,
    constraint_id: str | None = None,
    policy_decision_id: str | None = None,
) -> ConstraintExplanation:
    facts = [
        *_facts_for_spec(spec),
        f"query_actor({_q(actor_id)}).",
        f"query_state({_q(state)}).",
        f"query_action({_q(action)}).",
    ]
    program = "\n".join(
        [
            *facts,
            "candidate(T) :- query_state(S), query_action(A), transition(T,S,A,_).",
            'available(T) :- candidate(T), availability(T,"hard_available").',
            'available(T) :- candidate(T), availability(T,"possible_violation").',
            'blocked(T) :- candidate(T), availability(T,"hard_blocked").',
            'reason(T,"transition_matches_state_action") :- candidate(T).',
            'reason(T,"hard_available") :- candidate(T), availability(T,"hard_available").',
            'reason(T,"possible_violation") :- candidate(T), availability(T,"possible_violation").',
            'reason(T,"hard_blocked") :- blocked(T).',
            'reason(T,"norm_permitted") :- candidate(T), norm_status(T,"permitted").',
            'reason(T,"norm_forbidden") :- candidate(T), norm_status(T,"forbidden").',
            'reason(T,"norm_obligatory") :- candidate(T), norm_status(T,"obligatory").',
            'reason(T,"norm_discouraged") :- candidate(T), norm_status(T,"discouraged").',
            'violation(T,"forbidden_action") :- candidate(T), norm_status(T,"forbidden").',
            'obligation(T,A) :- candidate(T), norm_status(T,"obligatory"), transition(T,_,A,_).',
            "remediation(T,R) :- candidate(T), restorative_action(T,R).",
            'reason(T,"remediation_available") :- remediation(T,_).',
            "#show candidate/1.",
            "#show available/1.",
            "#show blocked/1.",
            "#show norm_status/2.",
            "#show reason/2.",
            "#show violation/2.",
            "#show obligation/2.",
            "#show remediation/2.",
        ]
    )
    symbols = _solve(program)
    candidates = [_string_arg(symbol, 0) for symbol in symbols if symbol.name == "candidate"]
    transition_id = candidates[0] if candidates else None
    available = any(symbol.name == "available" for symbol in symbols)
    hard_blocked = any(symbol.name == "blocked" for symbol in symbols)
    norm_status = NormStatus.UNKNOWN
    reasons = []
    violations = []
    obligations = []
    remediation_actions = []
    for symbol in symbols:
        if symbol.name == "norm_status" and transition_id == _string_arg(symbol, 0):
            norm_status = NormStatus(_string_arg(symbol, 1))
        if symbol.name == "reason" and transition_id == _string_arg(symbol, 0):
            reasons.append(_string_arg(symbol, 1))
        if symbol.name == "violation" and transition_id == _string_arg(symbol, 0):
            violations.append(_string_arg(symbol, 1))
        if symbol.name == "obligation" and transition_id == _string_arg(symbol, 0):
            obligations.append(_string_arg(symbol, 1))
        if symbol.name == "remediation" and transition_id == _string_arg(symbol, 0):
            remediation_actions.append(_string_arg(symbol, 1))
    if transition_id is None:
        reasons.append("no_transition_for_state_action")
    return ConstraintExplanation(
        constraint_id=constraint_id,
        policy_decision_id=policy_decision_id,
        actor_id=actor_id,
        state=state,
        action=action,
        transition_id=transition_id,
        available=available,
        hard_blocked=hard_blocked,
        blocked=hard_blocked,
        norm_status=norm_status,
        reasons=sorted(set(reasons)),
        obligations=sorted(set(obligations)),
        violations=sorted(set(violations)),
        remediation_actions=sorted(set(remediation_actions)),
        explanation_facts=sorted(set(reasons)),
    )


def _facts_for_spec(spec: IncentiveSpec) -> list[str]:
    facts: list[str] = []
    for state in spec.states.all:
        facts.append(f"state({_q(state)}).")
    for action in spec.actions.all:
        facts.append(f"action({_q(action)}).")
    for transition in spec.transitions:
        facts.extend(_facts_for_transition(transition))
    return facts


def _facts_for_transition(transition: Transition) -> list[str]:
    facts = [
        (
            f"transition({_q(transition.id)},{_q(transition.from_state)},"
            f"{_q(transition.action)},{_q(transition.to_state)})."
        ),
        f"availability({_q(transition.id)},{_q(transition.availability.value)}).",
        f"norm_status({_q(transition.id)},{_q(transition.norm_status.value)}).",
    ]
    if transition.enforcement is not None:
        facts.append(f"has_enforcement({_q(transition.id)}).")
        if transition.enforcement.restorative_action is not None:
            facts.append(
                f"restorative_action({_q(transition.id)},"
                f"{_q(transition.enforcement.restorative_action)})."
            )
    for tag in transition.tags:
        facts.append(f"tag({_q(transition.id)},{_q(tag)}).")
    return facts


def _solve(program: str) -> list[clingo.Symbol]:
    control = clingo.Control(["--warn=none"])
    control.add("base", [], program)
    control.ground([("base", [])])
    models: list[list[clingo.Symbol]] = []
    with control.solve(yield_=True) as handle:
        for model in handle:
            models.append(model.symbols(shown=True))
    return models[0] if models else []


def _q(value: str) -> str:
    return json.dumps(value)


def _string_arg(symbol: clingo.Symbol, index: int) -> str:
    argument = symbol.arguments[index]
    if argument.type is clingo.SymbolType.String:
        return argument.string
    return str(argument)


def _problem_message(code: str) -> str:
    if code == "hard_blocked_transition":
        return "hard-blocked transitions are rejected by the v1 constraint layer"
    if code == "auditable_forbidden_missing_enforcement":
        return "auditable forbidden transitions must define enforcement metadata"
    return code
