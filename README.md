# ICFRAME

ICFRAME compiles declarative incentive systems into deterministic, bounded-memory multi-agent experiments. IncentiveSpec v0.4 is a clean break from the legacy JSON/Mesa runtime.

## Quick Start

```bash
uv sync --group dev
uv run icframe packs
uv run icframe run public_goods --seed 7
```

Use `icframe study` for bounded Optuna studies and `icframe report` for self-contained
HTML exports. Run and study artifacts live under `.artifacts/icframe`;
`catalog.sqlite3` is only a rebuildable index.

## Optional Integrations

```bash
uv sync --extra symbolic   # Clingo at compile time
uv sync --extra optimize   # Optuna studies
uv sync --extra marl       # PettingZoo AEC and Parallel APIs
uv sync --extra llm        # Live model calls through LiteLLM
uv sync --extra analytics  # NetworkX artifact analysis
```

The base install contains only Pydantic. Mesa and the marimo viewer are removed.

For a live LLM domain, configure an OpenAI-compatible endpoint in `.env`:

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
