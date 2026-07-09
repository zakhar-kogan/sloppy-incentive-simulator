from __future__ import annotations

from icframe import PettingZooAECIncentiveEnv, load_incentive_spec


def main() -> None:
    spec = load_incentive_spec("incentive_spec_example_tokenmaxxing_v0_3.toml")
    env = PettingZooAECIncentiveEnv(spec)
    env.reset(seed=7)
    for _ in range(4):
        agent = env.agent_selection
        if agent is None:
            break
        mask = env.action_mask(agent)
        action = next(index for index, available in enumerate(mask) if available)
        env.step(action)
        print(agent, spec.actions.all[action], env.rewards[agent], env.infos[agent])


if __name__ == "__main__":
    main()
