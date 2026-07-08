from __future__ import annotations

from collections import defaultdict

from clingo import Symbol, SymbolType

from icframe.domain.norms import LawEvaluation


def _term_to_text(term: Symbol) -> str:
    if term.type is SymbolType.String:
        return term.string
    if term.type is SymbolType.Number:
        return str(term.number)
    return term.name


def parse_shown_symbols(symbols: list[Symbol]) -> LawEvaluation:
    buckets: dict[str, dict[str, set[str]]] = {
        "allowed": defaultdict(set),
        "forbidden": defaultdict(set),
        "violation": defaultdict(set),
    }

    for symbol in symbols:
        if symbol.name not in buckets or len(symbol.arguments) != 2:
            continue
        actor = _term_to_text(symbol.arguments[0])
        action = _term_to_text(symbol.arguments[1])
        buckets[symbol.name][actor].add(action)

    return LawEvaluation(
        allowed={actor: tuple(sorted(actions)) for actor, actions in buckets["allowed"].items()},
        forbidden={
            actor: tuple(sorted(actions)) for actor, actions in buckets["forbidden"].items()
        },
        violations={
            actor: tuple(sorted(actions)) for actor, actions in buckets["violation"].items()
        },
    )
