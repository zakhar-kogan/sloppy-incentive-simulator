# Technical post outline and evidence checklist

## 600-word post structure

1. **Problem (70 words):** KPI optimization can improve visible numbers while degrading trusted outcomes. Incentive changes need a reproducible stress test before deployment.
2. **Method (120 words):** IncentiveSpec, deterministic seeds, explicit matrix plans, trusted constraints, and the software-organization Goodhart experiment.
3. **Why serverless (110 words):** independent trials, deterministic shards, ephemeral CPU workers, bounded concurrency, retries, cancellation, and Object Storage handoff.
4. **Architecture (110 words):** controller, official SDK backend, versioned worker contract, MysteryBox secret injection, checksum manifest, atomic local import, rebuildable catalog.
5. **Measured result (130 words):** actual local/Nebius wall time, shard count, retry count, nominal KPI, trusted score, exploit rate, feasible/Pareto results, LLM tokens and cost. Insert only values produced by a retained run.
6. **Reproduction and limits (60 words):** public repository, worker digest, exact command, artifacts, known limitation that remote adaptive Optuna and partitioned LLM budgets are deferred.

## Evidence required before publishing

- [ ] Public repository URL and commit SHA.
- [ ] MIT `LICENSE` and worker `Dockerfile`.
- [ ] Container registry image digest.
- [ ] Redacted `icframe.toml` used for the run.
- [ ] Nebius Serverless Job IDs and screenshots/log exports.
- [ ] Object Storage completion marker and bundle SHA-256.
- [ ] Imported `summary.json`, `plan.json`, and `trials.jsonl`.
- [ ] Local-versus-Nebius comparison output for the same plan hash.
- [ ] Actual platform, preset, timeout, wall time, retries, and approximate cost.
- [ ] Token Factory provider/model, token counts, malformed/retry counts, and cost.
- [ ] Cancellation evidence from a second job.
- [ ] Desktop/tablet/mobile workbench screenshots and exported HTML report.
- [ ] Separate technical-post URL with at least 600 words.

Claims about scale, speed, or cost must point to captured run evidence. Marketing estimates do not count as measurements.
