from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from icframe.domain.incentive_spec import Archetype, PolicyKind
from icframe.llm import LLMClient, LLMRequest, UnknownLLMPricingError

from .types import ActionCandidate, Observation, PolicyChoice, PolicyFeedback


class Policy(Protocol):
    kind: PolicyKind

    def choose_action(
        self,
        observation: Observation,
        rng: random.Random,
    ) -> PolicyChoice: ...

    def learn(self, feedback: PolicyFeedback) -> dict[str, Any]: ...

    def snapshot(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class PolicyFactory:
    archetype: Archetype

    def create(self, llm_client: LLMClient | None) -> Policy | None:
        if self.archetype.policy is PolicyKind.EXTERNAL:
            return None
        return create_policy(self.archetype, llm_client)


@dataclass(slots=True)
class BasePolicy:
    kind: PolicyKind
    scalarizer: dict[str, float]
    config: dict[str, Any]

    def learn(self, feedback: PolicyFeedback) -> dict[str, Any]:
        del feedback
        return {}

    def snapshot(self) -> dict[str, Any]:
        return {}

    def expected(self, candidate: ActionCandidate) -> float:
        if "__scalar__" in candidate.visible_outcomes:
            return float(candidate.visible_outcomes["__scalar__"])
        return sum(
            self.scalarizer.get(channel, 0.0) * value
            for channel, value in candidate.visible_outcomes.items()
        )


@dataclass(slots=True)
class DeterministicPolicy(BasePolicy):
    def choose_action(self, observation: Observation, rng: random.Random) -> PolicyChoice:
        del rng
        if not observation.candidates:
            return PolicyChoice(failure="no_available_actions")
        preferences = self.config.get("preferences", [])
        for preferred in preferences:
            for candidate in observation.candidates:
                if candidate.action == preferred:
                    return PolicyChoice(action=candidate.action, target_id=candidate.target_id)
        scores = {candidate.key: self.expected(candidate) for candidate in observation.candidates}
        candidate = _best(observation.candidates, scores)
        return PolicyChoice(
            action=candidate.action,
            target_id=candidate.target_id,
            estimated_rewards=scores,
            probability=1.0,
            rationale="max_visible_scalar_reward",
        )


@dataclass(slots=True)
class StochasticWeightedPolicy(BasePolicy):
    def choose_action(self, observation: Observation, rng: random.Random) -> PolicyChoice:
        if not observation.candidates:
            return PolicyChoice(failure="no_available_actions")
        scores = {candidate.key: self.expected(candidate) for candidate in observation.candidates}
        candidate, probability = _weighted_choice(observation.candidates, scores, rng)
        return PolicyChoice(
            action=candidate.action,
            target_id=candidate.target_id,
            estimated_rewards=scores,
            probability=probability,
            rationale="weighted_visible_scalar_reward",
        )


@dataclass(slots=True)
class ValuePolicy(BasePolicy):
    counts: dict[str, int] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)

    def initial_value(self, candidate: ActionCandidate) -> float:
        if "initial_value" in self.config:
            return float(self.config["initial_value"])
        return self.expected(candidate)

    def estimates(self, candidates: tuple[ActionCandidate, ...]) -> dict[str, float]:
        return {
            candidate.key: self.values.get(candidate.key, self.initial_value(candidate))
            for candidate in candidates
        }

    def learn(self, feedback: PolicyFeedback) -> dict[str, Any]:
        key = _action_key(feedback.action, feedback.target_id)
        old_count = self.counts.get(key, 0)
        old_value = self.values.get(key, 0.0)
        count = old_count + 1
        value = old_value + (feedback.reward - old_value) / count
        self.counts[key] = count
        self.values[key] = value
        return {
            "action": key,
            "count": {"old": old_count, "new": count},
            "value": {"old": old_value, "new": value},
        }

    def snapshot(self) -> dict[str, Any]:
        return {"counts": dict(self.counts), "values": dict(self.values)}


@dataclass(slots=True)
class EpsilonGreedyPolicy(ValuePolicy):
    def choose_action(self, observation: Observation, rng: random.Random) -> PolicyChoice:
        if not observation.candidates:
            return PolicyChoice(failure="no_available_actions")
        epsilon = float(self.config.get("exploration_rate", 0.1))
        estimates = self.estimates(observation.candidates)
        if rng.random() < epsilon:
            candidate = rng.choice(observation.candidates)
            probability = epsilon / len(observation.candidates)
            rationale = "epsilon_explore"
        else:
            candidate = _best(observation.candidates, estimates)
            greedy = sum(
                1
                for item in observation.candidates
                if estimates[item.key] == estimates[candidate.key]
            )
            probability = (1.0 - epsilon) / greedy + epsilon / len(observation.candidates)
            rationale = "epsilon_exploit"
        return PolicyChoice(
            action=candidate.action,
            target_id=candidate.target_id,
            estimated_rewards=estimates,
            probability=probability,
            rationale=rationale,
        )


@dataclass(slots=True)
class UCBPolicy(ValuePolicy):
    def choose_action(self, observation: Observation, rng: random.Random) -> PolicyChoice:
        del rng
        if not observation.candidates:
            return PolicyChoice(failure="no_available_actions")
        unvisited = [item for item in observation.candidates if self.counts.get(item.key, 0) == 0]
        if unvisited:
            candidate = unvisited[0]
            scores = {item.key: math.inf for item in unvisited}
        else:
            total = max(sum(self.counts.values()), 1)
            coefficient = float(self.config.get("exploration_coefficient", 1.0))
            scores = {
                item.key: self.values.get(item.key, 0.0)
                + coefficient * math.sqrt(math.log(total + 1.0) / self.counts[item.key])
                for item in observation.candidates
            }
            candidate = _best(observation.candidates, scores)
        return PolicyChoice(
            action=candidate.action,
            target_id=candidate.target_id,
            estimated_rewards=scores,
            rationale="ucb",
        )


@dataclass(slots=True)
class GaussianThompsonPolicy(ValuePolicy):
    m2: dict[str, float] = field(default_factory=dict)

    def choose_action(self, observation: Observation, rng: random.Random) -> PolicyChoice:
        if not observation.candidates:
            return PolicyChoice(failure="no_available_actions")
        prior_variance = float(self.config.get("prior_variance", 1.0))
        samples = {}
        for item in observation.candidates:
            count = self.counts.get(item.key, 0)
            mean = self.values.get(item.key, self.initial_value(item))
            variance = (
                self.m2.get(item.key, 0.0) / max(count - 1, 1) if count > 1 else prior_variance
            )
            samples[item.key] = rng.gauss(mean, math.sqrt(max(variance, 1e-9) / (count + 1)))
        candidate = _best(observation.candidates, samples)
        return PolicyChoice(
            action=candidate.action,
            target_id=candidate.target_id,
            estimated_rewards=samples,
            rationale="gaussian_thompson_sample",
        )

    def learn(self, feedback: PolicyFeedback) -> dict[str, Any]:
        key = _action_key(feedback.action, feedback.target_id)
        old_count = self.counts.get(key, 0)
        old_mean = self.values.get(key, 0.0)
        old_m2 = self.m2.get(key, 0.0)
        count = old_count + 1
        delta = feedback.reward - old_mean
        mean = old_mean + delta / count
        m2 = old_m2 + delta * (feedback.reward - mean)
        self.counts[key] = count
        self.values[key] = mean
        self.m2[key] = m2
        return {"action": key, "count": count, "mean": mean, "m2": m2}

    def snapshot(self) -> dict[str, Any]:
        return {**ValuePolicy.snapshot(self), "m2": dict(self.m2)}


@dataclass(slots=True)
class ContextualPolicy(BasePolicy):
    weights: dict[str, dict[str, float]] = field(default_factory=dict)
    last_features: dict[str, dict[str, float]] = field(default_factory=dict)

    def choose_action(self, observation: Observation, rng: random.Random) -> PolicyChoice:
        if not observation.candidates:
            return PolicyChoice(failure="no_available_actions")
        scores = {}
        for item in observation.candidates:
            features = _features(observation, item)
            self.last_features[item.key] = features
            action_weights = self.weights.setdefault(item.key, {})
            scores[item.key] = sum(
                action_weights.get(name, 0.0) * value for name, value in features.items()
            )
        epsilon = float(self.config.get("exploration_rate", 0.1))
        if rng.random() < epsilon:
            candidate = rng.choice(observation.candidates)
            rationale = "contextual_explore"
        else:
            candidate = _best(observation.candidates, scores)
            rationale = "contextual_exploit"
        return PolicyChoice(
            action=candidate.action,
            target_id=candidate.target_id,
            estimated_rewards=scores,
            rationale=rationale,
        )

    def learn(self, feedback: PolicyFeedback) -> dict[str, Any]:
        key = _action_key(feedback.action, feedback.target_id)
        features = self.last_features.get(key, {"bias": 1.0})
        weights = self.weights.setdefault(key, {})
        prediction = sum(weights.get(name, 0.0) * value for name, value in features.items())
        error = feedback.reward - prediction
        rate = float(self.config.get("learning_rate", 0.1))
        changed = {}
        for name, value in features.items():
            old = weights.get(name, 0.0)
            new = old + rate * error * value
            weights[name] = new
            changed[name] = {"old": old, "new": new}
        return {"action": key, "prediction": prediction, "error": error, "weights": changed}

    def snapshot(self) -> dict[str, Any]:
        return {"weights": self.weights}


@dataclass(slots=True)
class QLearningPolicy(BasePolicy):
    q_values: dict[str, dict[str, float]] = field(default_factory=dict)

    def choose_action(self, observation: Observation, rng: random.Random) -> PolicyChoice:
        if not observation.candidates:
            return PolicyChoice(failure="no_available_actions")
        state_values = self.q_values.setdefault(observation.state, {})
        scores = {item.key: state_values.get(item.key, 0.0) for item in observation.candidates}
        epsilon = float(self.config.get("exploration_rate", 0.1))
        if rng.random() < epsilon:
            candidate = rng.choice(observation.candidates)
            rationale = "q_explore"
        else:
            candidate = _best(observation.candidates, scores)
            rationale = "q_exploit"
        return PolicyChoice(
            action=candidate.action,
            target_id=candidate.target_id,
            estimated_rewards=scores,
            rationale=rationale,
        )

    def learn(self, feedback: PolicyFeedback) -> dict[str, Any]:
        key = _action_key(feedback.action, feedback.target_id)
        alpha = float(self.config.get("learning_rate", 0.1))
        gamma = float(self.config.get("discount_factor", 0.9))
        current = self.q_values.setdefault(feedback.state, {})
        following = self.q_values.setdefault(feedback.next_state, {})
        old = current.get(key, 0.0)
        target = feedback.reward + gamma * max(following.values(), default=0.0)
        new = old + alpha * (target - old)
        current[key] = new
        return {"state": feedback.state, "action": key, "old": old, "new": new, "target": target}

    def snapshot(self) -> dict[str, Any]:
        return {"q_values": self.q_values}


@dataclass(slots=True)
class LLMPolicy(BasePolicy):
    client: LLMClient | None = None
    llm_config: Any = None

    def choose_action(self, observation: Observation, rng: random.Random) -> PolicyChoice:
        del rng
        if not observation.candidates:
            return PolicyChoice(failure="no_available_actions")
        if self.client is None:
            return PolicyChoice(failure="missing_llm_client")
        payload = {
            "state": observation.state,
            "resources": observation.resources,
            "actions": [
                {
                    "action": item.action,
                    "target_id": item.target_id,
                    "label": item.prompt_label,
                    "description": item.prompt_description,
                    "visible_outcomes": item.visible_outcomes,
                    "visible_sanctions": item.visible_sanctions,
                }
                for item in observation.candidates
            ],
            "history": observation.visible_history,
        }
        call_id = f"llm_{observation.observation_id}"
        request = LLMRequest(
            llm_call_id=call_id,
            policy_decision_id=f"decision_{observation.observation_id}",
            provider=self.llm_config.provider,
            model=self.llm_config.model,
            system_prompt=self.llm_config.system_prompt,
            prompt=json.dumps(payload, sort_keys=True),
            temperature=self.llm_config.temperature,
            require_json=self.llm_config.require_json,
            input_cost_per_million_tokens_usd=(self.llm_config.input_cost_per_million_tokens_usd),
            output_cost_per_million_tokens_usd=(self.llm_config.output_cost_per_million_tokens_usd),
        )
        started = time.perf_counter()
        try:
            response = self.client.complete(request)
        except UnknownLLMPricingError:
            raise
        except Exception as exc:
            return PolicyChoice(
                failure=f"llm_error:{type(exc).__name__}",
                llm_call={
                    "id": call_id,
                    "request_hash": request.request_hash,
                    "prompt": request.prompt,
                    "error": str(exc),
                    "status": "failed",
                    "failure_classification": type(exc).__name__,
                    "provider": request.provider,
                    "model": request.model,
                    "step": observation.step,
                    "agent_id": observation.agent_id,
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                    "estimated_cost": None,
                },
            )
        action = response.parsed.get(self.llm_config.action_field)
        target = response.parsed.get(self.llm_config.target_field)
        if not isinstance(action, str):
            failure = "malformed_llm_action"
        elif target is not None and not isinstance(target, str):
            failure = "malformed_llm_target"
        elif not any(
            item.action == action and item.target_id == target for item in observation.candidates
        ):
            failure = "invalid_llm_action"
        else:
            failure = None
        if response.error_type and failure is None:
            failure = f"llm_error:{response.error_type}"
        status = (
            "failed"
            if response.error_type
            else "completed"
            if failure is None
            else "invalid"
            if failure == "invalid_llm_action"
            else "malformed"
        )
        rationale = response.parsed.get("rationale") or response.parsed.get("reason")
        return PolicyChoice(
            action=action if isinstance(action, str) else None,
            target_id=target if isinstance(target, str) else None,
            rationale=rationale if isinstance(rationale, str) else None,
            failure=failure,
            llm_call={
                "id": call_id,
                "request_hash": request.request_hash,
                "prompt": request.prompt,
                "content": response.content,
                "parsed": response.parsed,
                "provider": response.provider,
                "model": response.model,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
                "estimated_cost": response.estimated_cost,
                "error": response.error_type,
                "status": status,
                "failure_classification": failure or response.error_type,
                "step": observation.step,
                "agent_id": observation.agent_id,
                "selected_action": action if isinstance(action, str) else None,
                "latency_ms": response.latency_ms,
                "retry_count": response.retry_count,
                "fallback_used": response.fallback_used,
            },
        )


def create_policy(archetype: Archetype, llm_client: LLMClient | None = None) -> Policy:
    common = {
        "kind": archetype.policy,
        "scalarizer": dict(archetype.scalarizer),
        "config": dict(archetype.policy_config),
    }
    if archetype.policy is PolicyKind.DETERMINISTIC:
        return DeterministicPolicy(**common)
    if archetype.policy is PolicyKind.STOCHASTIC_WEIGHTED:
        return StochasticWeightedPolicy(**common)
    if archetype.policy is PolicyKind.EPSILON_GREEDY:
        return EpsilonGreedyPolicy(**common)
    if archetype.policy is PolicyKind.UCB:
        return UCBPolicy(**common)
    if archetype.policy is PolicyKind.GAUSSIAN_THOMPSON:
        return GaussianThompsonPolicy(**common)
    if archetype.policy is PolicyKind.CONTEXTUAL:
        return ContextualPolicy(**common)
    if archetype.policy is PolicyKind.Q_LEARNING:
        return QLearningPolicy(**common)
    if archetype.policy is PolicyKind.LLM:
        return LLMPolicy(**common, client=llm_client, llm_config=archetype.llm)
    if archetype.policy is PolicyKind.EXTERNAL:
        raise ValueError("external policies are controlled through RuntimeEngine adapters")
    raise AssertionError(archetype.policy)


def _best(candidates: tuple[ActionCandidate, ...], scores: dict[str, float]) -> ActionCandidate:
    return max(enumerate(candidates), key=lambda item: (scores[item[1].key], -item[0]))[1]


def _weighted_choice(
    candidates: tuple[ActionCandidate, ...],
    scores: dict[str, float],
    rng: random.Random,
) -> tuple[ActionCandidate, float]:
    minimum = min(scores.values())
    weights = [scores[item.key] - minimum + 1e-6 for item in candidates]
    total = sum(weights)
    sample = rng.random() * total
    cursor = 0.0
    for candidate, weight in zip(candidates, weights, strict=True):
        cursor += weight
        if sample <= cursor:
            return candidate, weight / total
    return candidates[-1], weights[-1] / total


def _features(observation: Observation, candidate: ActionCandidate) -> dict[str, float]:
    features = {
        "bias": 1.0,
        f"state:{observation.state}": 1.0,
        f"action:{candidate.action}": 1.0,
    }
    for tag in candidate.tags:
        features[f"tag:{tag}"] = 1.0
    for channel, value in candidate.visible_outcomes.items():
        features[f"outcome:{channel}"] = float(value)
    return features


def _action_key(action: str, target_id: str | None) -> str:
    return f"{action}@{target_id}" if target_id else action
