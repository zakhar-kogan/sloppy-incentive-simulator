from __future__ import annotations

from clingo.control import Control

from icframe.domain.norms import LawEvaluation
from icframe.domain.scenario import Scenario
from icframe.ports.solver import SolverPort

from .parser import parse_shown_symbols

_ACTIONS = ("contribute", "withhold", "signal", "tamper")


class ClingoSolver(SolverPort):
    """Evaluate layered law programs with clingo."""

    def solve(self, scenario: Scenario) -> LawEvaluation:
        control = Control(arguments=["0"])
        control.add("base", [], self._build_program(scenario))
        control.ground([("base", [])])

        models: list[list] = []
        result = control.solve(on_model=lambda model: models.append(model.symbols(shown=True)))
        if not result.satisfiable:
            raise ValueError(f"law program for scenario {scenario.name!r} is unsatisfiable")
        if len(models) != 1:
            raise ValueError(
                "law program for scenario "
                f"{scenario.name!r} produced {len(models)} stable models; "
                "expected exactly one"
            )
        return parse_shown_symbols(models[0])

    def _build_program(self, scenario: Scenario) -> str:
        actor_facts = [f'actor("{agent.name}").' for agent in scenario.agents]
        topology_facts = [
            f'channel("{edge.source}","{edge.target}").'
            for edge in scenario.topology.materialize_edges(scenario.agent_names)
        ]
        action_facts = [f"action({action})." for action in _ACTIONS]
        sections = [*actor_facts, *topology_facts, *action_facts, scenario.laws.render()]
        return "\n".join(section for section in sections if section)
