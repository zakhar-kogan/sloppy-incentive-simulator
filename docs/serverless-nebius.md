# Nebius setup and reproducibility

ICFRAME v0.5 treats Nebius Serverless Jobs as an execution backend. The simulation engine and study aggregation do not import Nebius types; the backend adapter owns SDK calls, Object Storage transport, polling, retries, and cancellation.

## 1. Build and publish the worker

```bash
docker build -t cr.eu-north1.nebius.cloud/PROJECT/icframe-worker:0.5.0 .
docker push cr.eu-north1.nebius.cloud/PROJECT/icframe-worker:0.5.0
docker inspect --format='{{index .RepoDigests 0}}' cr.eu-north1.nebius.cloud/PROJECT/icframe-worker:0.5.0
```

Put the immutable digest, not a mutable tag, in `icframe.toml`. Copy `icframe.toml.example`, then set `parent_id`, `image`, `bucket`, `subnet_id`, and any Object Storage endpoint/profile required by your local AWS SDK configuration. Set `public_ip = true` when a remote LLM policy must reach Token Factory over HTTPS; otherwise keep it false and provide private egress separately.

## 2. Configure non-secret profiles

```bash
cp icframe.toml.example icframe.toml
export NEBIUS_API_KEY='...'
```

The controller reads normal local credentials through the official Nebius SDK configuration. Token Factory keys are read from `NEBIUS_API_KEY` locally. Remote jobs receive the key only through the configured MysteryBox secret reference. Set `remote_secret` to the real `mbsec-...` ID of a secret whose primary version contains the `NEBIUS_API_KEY` value; do not use a display name. The workbench, job request, orchestration log, and artifact bundle contain the profile name but never the key.

The default remote profile uses `cpu-d3`, `4vcpu-16gb`, a one-hour timeout, 32 trials per shard, four concurrent shards, ten-second polling, and three total attempts. Live-LLM studies are automatically collapsed to one shard and one in-flight job so a single budget guard owns all trial calls.

## 3. Run and inspect

```bash
# Local deterministic reference
uv run python scripts/flagship_goodhart.py --execution-profile local

# Submit the same plan to Nebius and wait for imported artifacts
uv run python scripts/flagship_goodhart.py --execution-profile nebius

# Run both and assert plan/trial equivalence after remote import
uv run python scripts/flagship_goodhart.py --execution-profile both

# Or use the generic CLI
uv run icframe study software_organization \
  --preset goodhart_audit \
  --execution-profile nebius
```

Every logical job is durable under `.artifacts/icframe/jobs/<id>/`:

- `manifest.json`: controller state and backend provenance;
- `orchestration.jsonl`: append-only lifecycle events;
- `plan.json`: immutable trial identities and hashes;
- `shards/*/attempt-*.json`: provider references and retry state;
- collected, checksum-verified bundles.

Imported run/study files under `.artifacts/icframe/runs` and `.artifacts/icframe/studies` are authoritative. `catalog.sqlite3` can be rebuilt with `uv run icframe catalog rebuild`.

## 4. Live smoke test checklist

The live test is intentionally opt-in because it creates billable resources:

1. Record the worker image digest and `icframe.toml` profile (without credentials).
2. Submit one CPU run and capture its Nebius job ID.
3. Run one budgeted Token Factory policy call and record provider/model/token usage.
4. Verify the completion marker, bundle SHA-256, imported summary, and replay/report output.
5. Submit a second job, cancel it, and verify the provider and local summary both become cancelled.
6. Save actual wall time, retries, resource preset, and provider cost evidence. Do not substitute estimates.

Use `NEBIUS_LIVE_TEST=1` only in a separately configured CI environment. The normal test suite uses local and fake backends and incurs no cloud cost.
