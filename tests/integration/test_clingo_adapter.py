from __future__ import annotations

import json
from pathlib import Path

from icframe.pipelines import load_scenario
from icframe.solvers.clingo import ClingoSolver


def test_clingo_adapter_matches_golden_law_projection() -> None:
    scenario = load_scenario("examples/microbenches/public_goods.json")
    expected = json.loads(Path("tests/golden/public_goods_laws.json").read_text())

    actual = ClingoSolver().solve(scenario).model_dump(mode="json")

    assert actual == expected
