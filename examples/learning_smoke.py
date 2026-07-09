from __future__ import annotations

from icframe import load_incentive_spec, run_incentive_simulation


def main() -> None:
    spec = load_incentive_spec("incentive_spec_example_learning_v0_3.toml")
    trace = run_incentive_simulation(spec, seed=11)
    agent = trace.final_agent_state["q_learner_000"]
    print(agent.memory["policy"])


if __name__ == "__main__":
    main()
