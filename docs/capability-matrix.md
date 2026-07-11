# Capability Matrix

| Area | v0.4 support | Rejected or external |
| --- | --- | --- |
| Scheduling | Sequential fixed, sequential random, atomic parallel | Staged schedules |
| State | Global values, agent resources and attributes | Unvalidated arbitrary mutation |
| Effects | Actor, target, population, all agents, global | Custom effect interpreters |
| Enforcement | Audit, detection, false positive/negative, sanctions, compliance rewards | Implicit norm behavior |
| Visibility | Full/local/prompt-only/no graph; numeric/scalar/label/hidden outcomes | Unknown observability modes |
| Metrics | Streaming sum, mean, count, rate, difference, ratio, weighted score | Arbitrary metric code in specs |
| Evaluation | Single or Pareto objectives; mean, median, worst, quantile seed reducers | Objectives outside trusted metrics |
| Policies | Deterministic, weighted stochastic, bandits, contextual, Q-learning, LLM, external | Training algorithms in core |
| Symbolic | Static compile-time availability and explanations | Per-turn Clingo calls |
| Retention | Audit, experiment, training | Unbounded in-memory traces |
| Reports | One internal UI/HTML projection with charts | Public report-model hierarchy or raw JSON presentation |

Unsupported enum values, selectors, fields, and legacy versions fail validation or compilation. They never degrade into no-ops.
