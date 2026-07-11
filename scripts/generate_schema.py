from __future__ import annotations

import json
from pathlib import Path

from icframe.domain.incentive_spec import DomainPackManifest, IncentiveSpec


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "docs"
    root.mkdir(exist_ok=True)
    (root / "incentive-spec-v0.4.schema.json").write_text(
        json.dumps(IncentiveSpec.model_json_schema(), indent=2, sort_keys=True) + "\n"
    )
    (root / "domain-pack-manifest-v0.4.schema.json").write_text(
        json.dumps(DomainPackManifest.model_json_schema(), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
