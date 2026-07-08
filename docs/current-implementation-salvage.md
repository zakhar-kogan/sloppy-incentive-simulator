# Current Implementation Salvage Notes

The current public-goods implementation is a useful prototype, not the v0.2 core.
The next implementation should be organized around `incentive_spec_v0_2.md` and
the tokenmaxxing TOML fixture.

## Preserve

- Strict Pydantic boundary models and deterministic JSON serialization.
- Seeded, reproducible simulation traces.
- Explicit separation between visible/proxy evaluation and trusted evaluation.
- Failure diagnostics for Goodhart gaming, reward hacking, collusion, and system hacking.
- NetworkX-style event projections for graph analytics.
- Artifact persistence and lightweight reporting as adapters over run traces.
- Optuna/search hooks as optional consumers of the canonical runtime.
- ASP/Clingo as a narrow constraint and explanation adapter.

## Replace

- Public-goods-specific scenario schema.
- Hardcoded actions such as `contribute`, `withhold`, `signal`, and `tamper`.
- Domain-specific state such as balances, public pools, and contribution counters.
- Incentive fields such as `contribution_bonus` and `withhold_penalty`.
- Runtime mechanics embedded directly in Mesa agent methods.
- Evaluation logic that assumes public-goods payoff semantics.

## v1 Boundary

The v1 core is the IncentiveSpec IR plus a Python transition runtime. Clingo should
validate constraints, explain availability/norm decisions, and report invariant
violations. It should not compute outcome vectors, stochastic enforcement, agent
policy, scalar rewards, state mutation, or metrics.

Richer social-law compilation, policy-as-code, counterfactual law edits, and formal
governance analysis belong in v2.
