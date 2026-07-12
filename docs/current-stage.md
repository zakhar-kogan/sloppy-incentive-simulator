# Current Project Stage

ICFRAME v0.4 is an integrated experimental simulator, not yet a general-purpose training framework.

## Implemented

- Multi-armed bandit policies: epsilon-greedy, UCB, Gaussian Thompson, and contextual learning
- Simple Q-learning for delayed reward experiments
- Sequential and atomic parallel multi-agent execution
- LLM policies with bounded visible history, redacted artifacts, replay, and study budgets
- External MARL control through PettingZoo AEC and Parallel APIs
- Guided public-goods, software-organization, and delayed-reward domain packs
- Interactive exact-value experiments, configurable study ranges, catalog history, comparisons, cancellation, charts, and report export
- Results-first Setup/Results workspaces with interpreted metrics, Mechanics projection,
  bounded agent statistics, and redacted per-run LLM inspection
- Single-objective and Pareto optimization across configurable seeds

## Deliberately Outside The Core

- MARL training algorithms and model checkpoints
- Distributed execution and remote job orchestration
- Arbitrary user code inside mechanics or policy selection
- Dynamic per-turn ASP solving
- Legacy Scenario/Mesa compatibility
- Executable social topology, typed builder, and graph-composer APIs
- An LLM-agent framework inside the simulator core

The current product stage is suitable for small audited experiments, local parameter studies, environment validation, and high-throughput bounded-memory episodes. Production-scale training orchestration is a later layer that should consume the stable core and PettingZoo contracts.
