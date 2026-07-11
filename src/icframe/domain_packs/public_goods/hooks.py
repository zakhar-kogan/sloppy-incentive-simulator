from __future__ import annotations

from icframe.core.hooks import (
    CommitContext,
    HookResult,
    InitContext,
    ResolvedStatePatch,
    StepContext,
)
from icframe.domain.incentive_spec import Operation


class PublicGoodsHooks:
    def initialize(self, context: InitContext) -> HookResult:
        del context
        return HookResult()

    def before_step(self, context: StepContext) -> HookResult:
        del context
        return HookResult()

    def after_commit(self, context: CommitContext) -> HookResult:
        contributors = sum(1 for event in context.events if event.action == "contribute")
        if not contributors or not context.after.agents:
            return HookResult()
        multiplier = float(context.hook_config.get("return_multiplier", 1.5))
        benefit = contributors * multiplier / len(context.after.agents)
        outcomes = {agent_id: {"agent.payoff": benefit} for agent_id in context.after.agents}
        patches = [
            ResolvedStatePatch(
                target=agent_id,
                field=("resources", "balance"),
                operation=Operation.ADD,
                value=benefit,
            )
            for agent_id in context.after.agents
        ]
        return HookResult(
            state_patches=patches,
            outcomes_by_agent=outcomes,
            global_outcome={"latent.social_welfare": contributors * multiplier},
        )

    def is_terminal(self, context: StepContext) -> bool:
        del context
        return False
