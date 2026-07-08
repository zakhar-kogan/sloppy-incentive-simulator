# Architecture

ICFRAME uses a strict hexagonal layout.

## Boundary rule
Pydantic models in `icframe.domain` are the only canonical package-boundary representation. External libraries stay behind adapters:
- Clingo stays inside `solvers/`
- Mesa stays inside `sim/`
- NetworkX stays inside `analytics/`
- Optuna stays inside `optimize/`

## Thin-slice flow
1. Load `Scenario`.
2. Materialize a layered `LawProgram` with the Clingo adapter.
3. Simulate a seedable public-goods world with the Mesa adapter.
4. Project the resulting event trace into interaction metrics.
5. Score the run with both visible and trusted objectives.
6. Optionally search over incentive parameters with the Optuna adapter.
7. Persist artifacts and provenance.

## Why visible and trusted scores are separate
A framework cannot claim to detect reward hacking or Goodhart behavior if the same mutable score both guides and judges the search loop. ICFRAME therefore keeps:
- a visible objective for agents and optimizers,
- a trusted adjudicator for acceptance.

## Deferred subsystems
The following are intentionally deferred until the thin slice is stable:
- live LLM codifiers and social agents
- BoTorch search
- ClingCon constraints
- dashboards and UI

The repository should add those only after they can be tested against the same trusted evaluator and golden fixtures.
