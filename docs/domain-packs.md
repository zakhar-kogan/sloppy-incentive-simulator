# Domain Packs

A domain pack is a directory containing `pack.toml`, an IncentiveSpec v0.4 TOML file, and optionally one narrow hook implementation.

`pack.toml` declares:

- a stable pack ID, title, description, and spec path;
- default single and Pareto objectives;
- guided parameters with descriptions, units, numeric steps, bounds or choices, defaults, and structured targets;
- an optional `module:factory` hook reference.

Parameter targets identify an entity type, entity ID where required, and field-path
segments. The same declarations drive exact workbench inputs, CLI overrides, and the
allowed Optuna search bounds. A study may narrow numeric bounds but cannot exceed the
pack manifest. Trusted constraints are not parameter targets and cannot enter a search
space.

## Reference Packs

- `public_goods`: parallel shared resources, externalities, sanctions, collusion pressure, and an `after_commit` public-return hook.
- `software_organization`: Goodhart behavior, audits, hidden outcomes, LLM policy decisions, and trusted quality/customer metrics.
- `delayed_reward_learning`: epsilon-greedy, UCB, Gaussian Thompson, contextual, and Q-learning behavior.

Run a pack with:

```bash
icframe run public_goods --param learners=4 --retention experiment
icframe study delayed_reward_learning --mode pareto --trials 40
```

Use `icframe packs` to list installed packs and the interactive UI for guided controls.

Typed Python builders and graph composition are deliberately deferred. Pack TOML is the
only domain-authoring contract in v0.4.
