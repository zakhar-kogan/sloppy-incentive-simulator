from __future__ import annotations

from collections import defaultdict

import mesa

from icframe.domain.events import Event, EventKind
from icframe.domain.norms import LawEvaluation
from icframe.domain.scenario import AgentConfig, AgentPolicy, Scenario
from icframe.domain.state import AgentSnapshot, SimulationTrace, WorldSnapshot
from icframe.ports.simulator import SimulatorPort


class PublicGoodsAgent(mesa.Agent):
    def __init__(
        self,
        model: PublicGoodsModel,
        config: AgentConfig,
        allowed_actions: tuple[str, ...],
        forbidden_actions: tuple[str, ...],
        targets: tuple[str, ...],
    ) -> None:
        super().__init__(model)
        self.name = config.name
        self.policy = config.policy
        self.balance = config.endowment
        self.payoff = 0.0
        self.allowed_actions = allowed_actions
        self.forbidden_actions = forbidden_actions
        self.targets = targets
        self.contributions = 0
        self.withholds = 0
        self.sent_messages = 0
        self.received_messages = 0
        self.violations = 0
        self.last_action: str | None = None

    def choose_action(self) -> str:
        tamper_probability = self.model.icframe_scenario.simulation.tamper_probability
        if self.policy is AgentPolicy.TAMPERER and self.model.random.random() < tamper_probability:
            return "tamper"

        if (
            self.policy is AgentPolicy.SIGNALER
            and self.model.current_step == 1
            and "signal" in self.allowed_actions
            and self.targets
        ):
            return "signal"

        if self.policy is AgentPolicy.OPPORTUNISTIC and "withhold" in self.allowed_actions:
            penalty = self.model.icframe_scenario.incentives.withhold_penalty
            bonus = self.model.icframe_scenario.incentives.contribution_bonus
            if penalty <= bonus:
                return "withhold"

        if "contribute" in self.allowed_actions:
            return "contribute"
        if "withhold" in self.allowed_actions:
            return "withhold"
        if self.allowed_actions:
            return self.allowed_actions[0]
        return "withhold"

    def step(self) -> None:
        action = self.choose_action()
        self.last_action = action
        self.model.apply_action(self, action)


class PublicGoodsModel(mesa.Model):
    def __init__(self, scenario: Scenario, laws: LawEvaluation, seed: int | None = None) -> None:
        run_seed = seed if seed is not None else scenario.simulation.seed
        super().__init__(rng=run_seed)
        self.icframe_scenario = scenario
        self.run_seed = run_seed
        self.laws = laws
        self.current_step = 0
        self.public_pool = 0.0
        self.events: list[Event] = []
        self.snapshots: list[WorldSnapshot] = []
        edges = scenario.topology.materialize_edges(scenario.agent_names)
        self.targets_by_source: dict[str, tuple[str, ...]] = defaultdict(tuple)
        grouped_targets: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            grouped_targets[edge.source].append(edge.target)
        for source, targets in grouped_targets.items():
            self.targets_by_source[source] = tuple(targets)

        for config in scenario.agents:
            allowed = tuple(sorted(laws.allowed.get(config.name, ("contribute", "withhold"))))
            forbidden = tuple(sorted(laws.forbidden.get(config.name, ())))
            PublicGoodsAgent(
                self,
                config,
                allowed_actions=allowed,
                forbidden_actions=forbidden,
                targets=self.targets_by_source.get(config.name, ()),
            )
        self.capture_snapshot(step=0)

    def apply_action(self, agent: PublicGoodsAgent, action: str) -> None:
        if action == "tamper":
            self._apply_tamper(agent)
            return
        if action == "signal":
            self._apply_signal(agent)
            return
        if action == "withhold":
            self._apply_withhold(agent)
            return
        self._apply_contribution(agent)

    def _apply_contribution(self, agent: PublicGoodsAgent) -> None:
        amount = self.icframe_scenario.simulation.contribution_amount
        reward = self.icframe_scenario.incentives.contribution_bonus
        agent.balance -= amount
        agent.balance += reward
        agent.payoff += reward - amount
        agent.contributions += 1
        self.public_pool += amount
        self.events.append(
            Event(
                step=self.current_step,
                actor=agent.name,
                kind=EventKind.CONTRIBUTE,
                amount=amount,
                reward=reward,
            )
        )

    def _apply_withhold(self, agent: PublicGoodsAgent) -> None:
        penalty = self.icframe_scenario.incentives.withhold_penalty
        agent.balance -= penalty
        agent.payoff -= penalty
        agent.withholds += 1
        self.events.append(
            Event(
                step=self.current_step,
                actor=agent.name,
                kind=EventKind.WITHHOLD,
                reward=-penalty,
            )
        )

    def _apply_signal(self, agent: PublicGoodsAgent) -> None:
        signal_cost = self.icframe_scenario.incentives.signal_cost
        coordination_bonus = self.icframe_scenario.incentives.coordination_bonus
        agent.balance -= signal_cost
        agent.payoff -= signal_cost
        if coordination_bonus:
            agent.balance += coordination_bonus
            agent.payoff += coordination_bonus
        targets = agent.targets or ()
        if not targets:
            self.events.append(
                Event(
                    step=self.current_step,
                    actor=agent.name,
                    kind=EventKind.SIGNAL,
                    reward=coordination_bonus - signal_cost,
                )
            )
            return
        for target_name in targets:
            target = self._agent_by_name(target_name)
            target.received_messages += 1
            agent.sent_messages += 1
            self.events.append(
                Event(
                    step=self.current_step,
                    actor=agent.name,
                    target=target_name,
                    kind=EventKind.SIGNAL,
                    reward=coordination_bonus - signal_cost,
                )
            )

    def _apply_tamper(self, agent: PublicGoodsAgent) -> None:
        forbidden = "tamper" in agent.forbidden_actions
        reward = self.icframe_scenario.incentives.contribution_bonus * 2 if not forbidden else 0.0
        penalty = self.icframe_scenario.incentives.tamper_penalty if forbidden else 0.0
        agent.balance += reward
        agent.balance -= penalty
        agent.payoff += reward - penalty
        tags = ("reward_hack",) if reward else ()
        self.events.append(
            Event(
                step=self.current_step,
                actor=agent.name,
                kind=EventKind.TAMPER,
                reward=reward - penalty,
                tags=tags,
            )
        )
        if forbidden:
            agent.violations += 1
            violation_penalty = self.icframe_scenario.incentives.violation_penalty
            agent.balance -= violation_penalty
            agent.payoff -= violation_penalty
            self.events.append(
                Event(
                    step=self.current_step,
                    actor=agent.name,
                    kind=EventKind.VIOLATION,
                    reward=-violation_penalty,
                    metadata={"action": "tamper"},
                )
            )

    def _agent_by_name(self, name: str) -> PublicGoodsAgent:
        for agent in self.agents:
            if isinstance(agent, PublicGoodsAgent) and agent.name == name:
                return agent
        raise KeyError(name)

    def _distribute_public_pool(self) -> None:
        if not self.agents:
            return
        multiplier = self.icframe_scenario.simulation.public_return_multiplier
        share = (self.public_pool * multiplier) / len(self.agents)
        for agent in self.agents:
            if not isinstance(agent, PublicGoodsAgent):
                continue
            agent.balance += share
            agent.payoff += share
        self.public_pool = 0.0

    def capture_snapshot(self, step: int) -> None:
        snapshots = [
            AgentSnapshot(
                name=agent.name,
                balance=agent.balance,
                payoff=agent.payoff,
                contributions=agent.contributions,
                withholds=agent.withholds,
                sent_messages=agent.sent_messages,
                received_messages=agent.received_messages,
                violations=agent.violations,
                last_action=agent.last_action,
            )
            for agent in sorted(self.agents, key=lambda item: item.name)
            if isinstance(agent, PublicGoodsAgent)
        ]
        self.snapshots.append(
            WorldSnapshot(
                step=step,
                public_pool=self.public_pool,
                allowed_actions={
                    name: tuple(sorted(actions)) for name, actions in self.laws.allowed.items()
                },
                agents=snapshots,
            )
        )

    def run(self) -> SimulationTrace:
        for step in range(1, self.icframe_scenario.simulation.rounds + 1):
            self.current_step = step
            self.agents.shuffle_do("step")
            self._distribute_public_pool()
            self.capture_snapshot(step=step)
        return SimulationTrace(
            scenario_name=self.icframe_scenario.name,
            seed=self.run_seed,
            events=self.events,
            snapshots=self.snapshots,
        )


class MesaSimulator(SimulatorPort):
    def run(
        self,
        scenario: Scenario,
        laws: LawEvaluation,
        seed: int | None = None,
    ) -> SimulationTrace:
        return PublicGoodsModel(scenario=scenario, laws=laws, seed=seed).run()
