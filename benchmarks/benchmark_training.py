from __future__ import annotations

import argparse
import json
import time
import tracemalloc

from icframe.core import RuntimeEngine, compile_runtime, load_domain_pack
from icframe.core.observer import NoopObserver
from icframe.core.packs import apply_parameters
from icframe.domain.incentive_spec import RetentionProfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    args = parser.parse_args()
    pack = apply_parameters(load_domain_pack("delayed_reward_learning"), {"steps": args.steps})
    engine = RuntimeEngine(
        compile_runtime(pack),
        run_id="benchmark-training",
        seed=11,
        observer=NoopObserver(),
        retention=RetentionProfile.TRAINING,
    )
    tracemalloc.start()
    started = time.perf_counter()
    summary = engine.run()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert summary.checkpoints == []
    assert all(
        len(agent.history) <= engine.plan.visibility[agent.visibility_profile].history_events
        for agent in engine.world.agents.values()
    )
    print(
        json.dumps(
            {
                "steps": summary.steps_completed,
                "agent_turns": summary.event_count,
                "seconds": elapsed,
                "agent_turns_per_second": summary.event_count / elapsed,
                "peak_traced_bytes": peak,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
