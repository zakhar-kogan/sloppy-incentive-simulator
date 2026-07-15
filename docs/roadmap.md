# Deferred Roadmap

The following initiatives are intentionally outside v0.4.1. They are recorded here so the
results release can remain small without losing the longer product direction.

## Executable Topology

**Motivation:** Represent directed `observe`, `target`, and `communicate` capabilities between
agent groups and enforce them during candidate generation and prompt projection.

**Dependencies:** Stable v0.4 mechanics projection, explicit compatibility rules, and at least
one real experiment whose result changes under connectivity constraints.

**Entry criterion:** Begin only when an experiment requires connectivity semantics that cannot
be expressed through populations, targeted transitions, and visibility profiles.

## Combined Domain Graph

**Motivation:** Add agent connectivity as a layer in the existing Mechanics experience while
keeping mechanics and capability edge semantics visually distinct.

**Dependencies:** Executable topology and evidence that users need both layers simultaneously.

**Entry criterion:** Begin after topology artifacts exist and the combined view can be tested
against real runs. Do not introduce a generic graph framework solely for visualization.

## Typed Python Authoring

**Motivation:** Provide validated policy configurations, immutable in-memory domain packs, and
an `icframe.config` API that compiles to the canonical IncentiveSpec runtime.

**Dependencies:** Stable topology decisions, typed policy configuration models, serialization
rules, and clear worker restrictions for in-memory hooks.

**Entry criterion:** Begin after concrete TOML authoring friction is collected from v0.4.1
users. Preserve file-backed packs as the reproducible interchange format.

## In-Memory Multiprocess Studies

**Motivation:** Run authored in-memory packs across worker processes.

**Dependencies:** Importable hook references or a safe serialization contract and the Python
authoring API.

**Entry criterion:** Begin only when single-worker studies are a measured bottleneck.

## Fleet Operations And Spend

**Motivation:** Cross-run LLM spend dashboards, organization monitoring, and remote execution.

**Dependencies:** Stable per-run usage artifacts, authentication, durable remote job state,
and an explicit deployment model.

**Entry criterion:** Begin when ICFRAME moves beyond a local single-author workbench.
