# ICFRAME

ICFRAME compiles declarative incentive systems into deterministic, bounded-memory multi-agent experiments. v0.5 adds portable study plans and pluggable local or serverless execution without changing IncentiveSpec v0.4 or the deterministic engine.

## Quick Start

```bash
uv sync --group dev
uv run icframe packs
uv run icframe run public_goods --seed 7
uv run icframe study software_organization --preset goodhart_audit
uv run icframe ui
```

Open `http://127.0.0.1:8765`. The v0.5 workbench keeps Setup independent from Results and
supports exact experiment parameters, seed batches, deterministic matrix/random studies, interpreted
metrics, independent charts, exercised-mechanics inspection, agent statistics, redacted LLM
calls, comparisons, cancellation, and self-contained report export. Run and study artifacts
live under `.artifacts/icframe`; `catalog.sqlite3` is only a rebuildable index.

Execution and LLM connections are selected by name from a versioned `icframe.toml`. Local execution remains the default. Nebius Serverless Jobs is the first remote backend; Nebius Token Factory is an OpenAI-compatible preset behind the existing provider-neutral LLM client. The browser never receives configured cloud credentials.

```bash
cp icframe.toml.example icframe.toml
uv sync --extra nebius --extra llm
uv run icframe study software_organization \
  --preset goodhart_audit \
  --execution-profile nebius
```

See the [Nebius setup and reproducibility guide](docs/serverless-nebius.md), [v0.5 architecture](docs/architecture.md), and [challenge evidence checklist](docs/challenge-post-outline.md).

The workbench also supports live selectable runs, validated population composition,
evidence-linked findings, and parameter quick values.

LLM base URL, default model, temperature, and default prompt are saved in versioned browser
storage; API keys remain browser-session-only. Domain packs provide population templates and
evidence-backed causal Mechanics flows alongside the exact executable state machine.

## Optional Integrations

```bash
uv sync --extra symbolic   # Clingo at compile time
uv sync --extra optimize   # Optuna studies
uv sync --extra marl       # PettingZoo AEC and Parallel APIs
uv sync --extra llm        # Live model calls through LiteLLM
uv sync --extra analytics  # NetworkX artifact analysis
uv sync --extra nebius     # Nebius Serverless Jobs and Object Storage
```

The base install contains only Pydantic. Mesa and the marimo viewer are removed.

For a live LLM domain, configure an OpenAI-compatible endpoint in `.env` or enter
session-only credentials in the workbench:

```bash
ICFRAME_LLM_BASE_URL=https://api.openai.com/v1
ICFRAME_LLM_API_KEY=replace-me
ICFRAME_LLM_MODEL=openai/gpt-4o-mini
```

There is no fake LLM product mode. Deterministic model doubles exist only in tests,
and replay reads recorded parsed responses from run artifacts.

## Python API

```python
from icframe import RunConfig, load_domain_pack, run_experiment

pack = load_domain_pack("public_goods")
summary = run_experiment(pack, RunConfig(seed=7))
print(summary.metrics)
```

`notebooks/library_quickstart.py` is a dependency-free, cell-oriented example that
runs through the same public API in a Python notebook, Jupyter editor, or marimo.

See [architecture](docs/architecture.md), [capabilities](docs/capability-matrix.md), [domain packs](docs/domain-packs.md), and [current stage](docs/current-stage.md).
