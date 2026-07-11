# Threat model

## Primary failure classes
- **Goodhart gaming**: the optimized proxy improves while the trusted score degrades.
- **Reward hacking**: agents obtain reward through loopholes in the reward mapping.
- **Collusion**: harmful reciprocal concentration persists above baseline topology expectations.
- **System hacking**: logs, evaluator state, or communication channels are tampered with.

## v0.4 assumptions
- The trusted adjudicator is outside the optimizer's search space.
- Core tests must run without live model calls.
- Runtime and hook hashes make replay drift explicit.
- Artifact files are authoritative; the local catalog can be discarded and rebuilt.

## Known non-goals
- No claim of full prompt-to-ASP semantic equivalence.
- No claim that LLM-agent behavior is evidence of real social behavior.
- No protection against malicious trusted Python hook packages installed by the operator.
- No hard guarantee that a provider-reported LLM cost cannot exceed a remaining budget on its final call.

## Consequence for implementation
Every new subsystem must answer three questions before it joins the core path:
1. What trusted signal judges it?
2. What deterministic fixture proves it behaves as intended?
3. What retained input is sufficient to replay or explain it?
