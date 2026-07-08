# Threat model

## Primary failure classes
- **Goodhart gaming**: the optimized proxy improves while the trusted score degrades.
- **Reward hacking**: agents obtain reward through loopholes in the reward mapping.
- **Collusion**: harmful reciprocal concentration persists above baseline topology expectations.
- **System hacking**: logs, evaluator state, or communication channels are tampered with.

## MVP assumptions
- The trusted adjudicator is outside the optimizer's search space.
- Core tests must run without live model calls.
- Reproducibility matters more than behavioral richness in the first release.

## Known non-goals
- No claim of full prompt-to-ASP semantic equivalence.
- No claim that LLM-agent behavior is evidence of real social behavior.
- No UI-first workflow.

## Consequence for implementation
Every new subsystem must answer two questions before it joins the core path:
1. What trusted signal judges it?
2. What deterministic fixture proves it behaves as intended?
