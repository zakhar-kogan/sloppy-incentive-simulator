from __future__ import annotations

from dataclasses import replace

from icframe.core import RuntimeEngine, compile_runtime, load_domain_pack
from icframe.core.observer import NoopObserver
from icframe.domain.incentive_spec import IncentiveSpec


def test_clingo_compiles_availability_and_explanations_once(monkeypatch) -> None:
    pack = load_domain_pack("public_goods")
    payload = pack.spec.model_dump(mode="python", by_alias=True)
    payload["symbolic"] = {
        "enabled": True,
        "rules": [
            'blocked(T) :- tag(T,"tamper").',
            'reason(T,"tamper_rule") :- tag(T,"tamper").',
        ],
    }
    plan = compile_runtime(replace(pack, spec=IncentiveSpec.model_validate(payload)))
    transition = plan.transitions_by_state_action[("active", "tamper")]
    assert transition.availability.value == "hard_blocked"
    assert "tamper_rule" in transition.explanation_reasons

    def no_per_turn_solver(spec):  # pragma: no cover - called only on regression
        raise AssertionError(f"solver invoked during execution for {spec.spec.name}")

    monkeypatch.setattr("icframe.symbolic.compile_symbolic", no_per_turn_solver)
    summary = RuntimeEngine(
        plan,
        run_id="symbolic-once",
        seed=7,
        observer=NoopObserver(),
    ).run()
    assert summary.steps_completed == pack.spec.experiment.steps
