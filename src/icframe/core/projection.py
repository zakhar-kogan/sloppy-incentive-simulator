from __future__ import annotations

from dataclasses import dataclass

from icframe.domain.incentive_spec import Availability, OutcomeVector, OutcomeVisibility

from .compiler import RuntimePlan
from .execution import apply_effect
from .types import (
    ActionCandidate,
    AgentState,
    CompiledTransition,
    Observation,
    WorldSnapshot,
)


@dataclass(frozen=True, slots=True)
class ObservationProjector:
    """Project agent-visible state from an immutable world snapshot."""

    plan: RuntimePlan

    def project(
        self,
        *,
        run_id: str,
        agent_id: str,
        snapshot: WorldSnapshot,
        agent_state: AgentState,
    ) -> Observation:
        agent = snapshot.agents[agent_id]
        profile = self.plan.visibility[agent_state.visibility_profile]
        visible_candidates = tuple(
            ActionCandidate(
                transition_id=transition.id,
                action=transition.action,
                target_id=target_id,
                norm_status=transition.norm_status,
                tags=transition.tags,
                visible_outcomes=self._visible_effects(
                    transition.effects,
                    agent_id,
                    target_id,
                    snapshot,
                    profile.outcomes,
                    agent_state.scalarizer,
                ),
                visible_sanctions=self._visible_effects(
                    transition.enforcement.sanctions if transition.enforcement else (),
                    agent_id,
                    target_id,
                    snapshot,
                    profile.sanctions,
                    agent_state.scalarizer,
                ),
                prompt_label=transition.prompt_label if profile.prompts else None,
                prompt_description=(
                    transition.prompt_description if profile.prompts else None
                ),
            )
            for transition, target_id in self.candidates(agent_id, snapshot)
        )
        history_count = profile.history_events
        history = tuple(agent_state.history)[-history_count:] if history_count else ()
        return Observation(
            observation_id=f"obs_{run_id}_{snapshot.step:08d}_{agent_id}",
            run_id=run_id,
            step=snapshot.step,
            agent_id=agent_id,
            state=agent.state,
            resources=dict(agent.resources),
            candidates=visible_candidates,
            visible_history=history,
        )

    def candidates(
        self,
        agent_id: str,
        snapshot: WorldSnapshot,
    ) -> list[tuple[CompiledTransition, str | None]]:
        state = snapshot.agents[agent_id].state
        result = []
        for transition in self.plan.transitions_by_state.get(state, ()):
            if transition.availability is Availability.HARD_BLOCKED:
                continue
            if not transition.requires_target:
                result.append((transition, None))
                continue
            for target_id, target in sorted(snapshot.agents.items()):
                if target_id == agent_id:
                    continue
                if (
                    transition.target_populations
                    and target.population not in transition.target_populations
                ):
                    continue
                result.append((transition, target_id))
        return result

    @staticmethod
    def _visible_effects(
        effects,
        agent_id: str,
        target_id: str | None,
        snapshot: WorldSnapshot,
        visibility: OutcomeVisibility,
        scalarizer: dict[str, float],
    ) -> OutcomeVector:
        if visibility in {OutcomeVisibility.HIDDEN, OutcomeVisibility.LABEL_ONLY}:
            return {}
        outcomes: dict[str, OutcomeVector] = {}
        global_outcome: OutcomeVector = {}
        for effect in effects:
            apply_effect(
                effect,
                snapshot,
                agent_id,
                target_id,
                outcomes,
                global_outcome,
            )
        own = dict(outcomes.get(agent_id, {}))
        if visibility is OutcomeVisibility.FULL_NUMERIC:
            for channel, value in global_outcome.items():
                own[channel] = own.get(channel, 0.0) + value
            return own
        return {
            "__scalar__": sum(
                scalarizer.get(channel, 0.0) * value for channel, value in own.items()
            )
        }
