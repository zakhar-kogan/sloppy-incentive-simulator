from __future__ import annotations

import json
from dataclasses import dataclass, field

from icframe.domain.incentive_spec import IncentiveSpec


@dataclass(slots=True)
class SymbolicCompilation:
    blocked: set[str] = field(default_factory=set)
    reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)


def compile_symbolic(spec: IncentiveSpec) -> SymbolicCompilation:
    """Compile optional static ASP rules once.

    Rules may derive ``blocked(Transition)`` and ``reason(Transition, Text)``.
    Dynamic per-turn symbolic queries are intentionally outside v0.4.
    """

    try:
        import clingo
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise RuntimeError(
            "this domain pack declares symbolic rules; install icframe[symbolic]"
        ) from exc

    facts = []
    for transition in spec.transitions:
        facts.extend(
            [
                f"transition({_quote(transition.id)}).",
                f"from_state({_quote(transition.id)},{_quote(transition.from_state)}).",
                f"action({_quote(transition.id)},{_quote(transition.action)}).",
                f"norm({_quote(transition.id)},{_quote(transition.norm_status.value)}).",
            ]
        )
        for tag in transition.tags:
            facts.append(f"tag({_quote(transition.id)},{_quote(tag)}).")
    program = "\n".join([*facts, *spec.symbolic.rules, "#show blocked/1.", "#show reason/2."])
    control = clingo.Control(["--warn=none"])
    control.add("base", [], program)
    control.ground([("base", [])])
    symbols = []
    with control.solve(yield_=True) as handle:
        for model in handle:
            symbols = model.symbols(shown=True)
            break
    blocked: set[str] = set()
    reasons: dict[str, list[str]] = {}
    for symbol in symbols:
        if symbol.name == "blocked":
            blocked.add(_arg(symbol, 0))
        elif symbol.name == "reason":
            reasons.setdefault(_arg(symbol, 0), []).append(_arg(symbol, 1))
    return SymbolicCompilation(
        blocked=blocked,
        reasons={key: tuple(sorted(set(values))) for key, values in reasons.items()},
    )


def _quote(value: str) -> str:
    return json.dumps(value)


def _arg(symbol, index: int) -> str:
    value = symbol.arguments[index]
    return value.string if value.type.name == "String" else str(value)
