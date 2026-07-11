"""Notebook-friendly ICFRAME quickstart using only the public library API."""

# %%
from pathlib import Path

from icframe import RunConfig, load_domain_pack, run_experiment


# %%
def run_demo(artifact_root: str | Path = ".artifacts/icframe"):
    pack = load_domain_pack("software_organization")
    config = RunConfig(
        seed=19,
        parameters={"steps": 12, "audit_probability": 0.35},
        artifact_root=Path(artifact_root),
    )
    return run_experiment(pack, config)


# %%
if __name__ == "__main__":
    summary = run_demo()
    print(summary.metrics)
