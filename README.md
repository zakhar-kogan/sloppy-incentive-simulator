# ICFRAME

ICFRAME is a reproducible incentive-stress-testing research framework. This repository treats the problem as systems engineering first: typed governance artifacts, solver-backed law evaluation, deterministic simulation, explicit provenance, and an optimizer loop that is judged by a trusted evaluator instead of only by the proxy reward it optimizes.

That positioning is deliberate. The strongest version of the original concept is not a claim to a new formal mechanism-design theory. It is a claim that we can build better infrastructure for stress-testing incentive schemes under adversarial pressure, with enough structure to reproduce results and enough instrumentation to distinguish outcome classes that are usually conflated.

## Concept critique

### What survives scrutiny
- Symbolic norms plus agent-based simulation plus provenance is a useful combination.
- Separating hard rules from LLM codification is directionally correct.
- Communication topology is a valid experimental factor.
- Versioned governance artifacts and run outputs materially improve reproducibility.

### What needed correction
1. Novelty claims had to be reduced. This repository is an integration and evaluation framework, not a new general theory.
2. A norm hierarchy alone does not create a formal Goodhart boundary. Optimization can still satisfy the proxy while degrading the real objective.
3. Prompt-to-ASP semantic equivalence is not a credible routine governance gate. Natural-language prompts are not precise normative anchors, and strong equivalence checking in ASP is too expensive and fragile for the default workflow.
4. Failure modes must be operationally separated. Outcome summaries alone do not distinguish Goodhart gaming, reward hacking, collusion, and system hacking.
5. LLM agents are scientifically fragile as primary empirical evidence. In this repository they are deferred behind plugin boundaries.
6. Collusion cannot be defined only by explicit messaging. The framework keeps topology as a factor, not the center of the novelty story.
7. If the evaluator is mutable inside the search loop, the system can optimize the evaluator instead of the intended goal. ICFRAME therefore splits visible and trusted evaluation.

## Current v1 concept as ASCII chart

```text
                         ICFRAME v1 (current concept)

+--------------------------------------------------------------------------------+
| Immutable anchors                                                              |
|--------------------------------------------------------------------------------|
| 1. Original prompt  --> semantic intent anchor                                 |
| 2. Hard rules       --> human-authored, human-validated social laws            |
+--------------------------------------------------------------------------------+

        | codify / inject                               | compare back
        v                                               ^
+--------------------+      +------------------------+  |
| Codifier pipeline  | ---> | ASP law program        |--+
|--------------------|      |------------------------|
| NL prompt          |      | Layer 1: hard rules    |
| DSPy + Pydantic    |      | Layer 2: moral laws    |
| human validation   |      | Layer 3: incentives    |
+--------------------+      +------------------------+
          |                              |
          | world constraints            | governance-time diffs
          v                              v
+--------------------------------------------------------------------------------+
| Simulation loop                                                                 |
|--------------------------------------------------------------------------------|
| Mesa populations:                                                               |
| - bandit agents                                                                  |
| - RL agents                                                                      |
| - tier-1 LLM agents                                                              |
| - tier-2 LLM agents                                                              |
|                                                                                  |
| Communication topology (versioned): isolated / local_k / broadcast / coalition   |
| GM narration template feeds LLM-facing world state                               |
+--------------------------------------------------------------------------------+
          |
          v
+-----------------------------+
| Evaluation                  |
|-----------------------------|
| loss = undesirable tx       |
|      + inequality (Gini)    |
|      + efficiency           |
| graph analytics for         |
| collusion/exploit routes    |
+-----------------------------+
          |
          v
+-----------------------------+
| Optimisation                |
|-----------------------------|
| BoTorch / Optuna proposes   |
| mutations to laws, params,  |
| topology, penalties, loss   |
| weights, operator weights   |
+-----------------------------+
          |
          v
+-----------------------------+
| Drift check                 |
|-----------------------------|
| re-derive ASP from prompt   |
| compare semantics to        |
| current law program         |
+-----------------------------+
```

## Design corrections implemented here

### 1. Visible objective vs trusted adjudicator
ICFRAME exposes a visible objective that agents or optimization routines can target, and a separate trusted adjudicator that decides whether a candidate is genuinely better. That separation is what lets the framework label Goodhart-style failures honestly.

### 2. Operational failure-mode labels
The core domain model includes explicit diagnostics for:
- Goodhart gaming: proxy score improves while trusted score degrades.
- Reward hacking: reward is achieved through loophole-like events.
- Collusion: concentrated reciprocal coordination and harmful action concentration exceed thresholds.
- System hacking: unauthorized tampering with evaluator, logs, or channels is observed.

### 3. Executable conformance instead of grand semantic promises
The first build uses golden scenarios, deterministic fixtures, and regression traces. That is a credible basis for governance conformance; full semantic equivalence is not.

### 4. Typed core, optional live integrations
Pydantic domain models are the package boundary. Mesa, Clingo, NetworkX, and Optuna stay behind ports and adapters. Live LLM codification is intentionally excluded from the verified core.

### 5. Staged benchmark ladder
1. Spec and evaluator conformance tests
2. Goodhart and reward microbenchmarks
3. Collusion benchmark with ablations
4. System-hacking benchmark
5. Public-goods regression
6. Insider-information regression
7. Combined stress tests

## Research grounding
- AI Economist — https://arxiv.org/abs/2108.02755
- Adaptive Incentive Design with Multi-Agent Meta-Gradient RL — https://ifaamas.org/Proceedings/aamas2022/pdfs/p1436.pdf
- Goodhart’s Law in Reinforcement Learning — https://arxiv.org/abs/2310.09144
- Categorizing Variants of Goodhart’s Law — https://arxiv.org/abs/1803.04585
- Defining and Characterizing Reward Hacking — https://arxiv.org/abs/2209.13085
- AI Safety Gridworlds — https://arxiv.org/abs/1711.09883
- Strong and Uniform Equivalence in ASP — https://www.kr.tuwien.ac.at/projects/eq/aaai05.pdf
- Society-in-the-Loop — https://arxiv.org/abs/1707.07232
- Information-Theoretic Collusion Detection in Multi-Agent Games — https://proceedings.mlr.press/v180/bonjour22a/bonjour22a.pdf
- Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems — https://arxiv.org/abs/2602.15198

## Repository shape

```text
src/icframe/
├── analytics/networkx/     # Graph projections and collusion metrics
├── cli/                    # Thin command-line entrypoints
├── domain/                 # Canonical typed models
├── optimize/optuna/        # Search adapter for incentive mutation
├── pipelines/              # Thin orchestration over ports
├── ports/                  # Hexagonal interfaces
├── sim/mesa/               # Deterministic simulation backend
└── solvers/clingo/         # Law evaluation backend
```

## Verified thin slice in this repository
The first implementation session ships a runnable thin slice:
- strict Pydantic domain models
- a Clingo adapter that materializes allowed and forbidden actions from a layered law program
- a Mesa public-goods micro-world with deterministic seeds
- a NetworkX projection for interaction and collusion metrics
- a trusted evaluator that separates proxy reward from adjudication
- an Optuna adapter that searches incentive parameters
- integration and end-to-end tests that exercise the full path

## Getting started

```bash
uv sync --group dev
uv run pytest
uv run icframe optimize examples/microbenches/public_goods.json --trials 5 --seed 7 --output-dir .artifacts/demo
```

The command writes simulation outputs and provenance to the chosen output directory.

Each run now also writes `summary.json`, which contains chart-ready aggregates: event counts, per-agent outcomes, step summaries, and graph edges.

## Viewing experiment results

> The repository now includes a marimo app at `notebooks/experiment_viewer.py` for inspecting persisted runs.

```bash
uv sync --group dev --group viz
uv run icframe optimize examples/microbenches/public_goods.json --trials 5 --seed 7 --output-dir .artifacts/demo
uv run --group viz marimo edit notebooks/experiment_viewer.py
```

> If you prefer a one-shot app server instead of the editor, run `uv run --group viz marimo run notebooks/experiment_viewer.py`.

The viewer shows:

- visible vs trusted score and score gap
- contribution, inequality, throughput, reciprocity, and collusion metrics
- event mix charts
- per-agent balance and payoff charts
- step-by-step system trajectory
- interaction graph rendering and edge tables


### Static HTML report

For a lightweight shareable artifact with inline charts, generate a self-contained HTML report from any persisted run:

```bash
uv run icframe report .artifacts/demo
open .artifacts/demo/report.html
```

The report includes:

- headline metrics and diagnostics
- event-count bar chart
- final balance/payoff bar charts
- balance/payoff trajectory charts
- interaction graph SVG
- tabular summaries for agents, steps, and graph edges