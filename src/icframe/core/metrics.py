from __future__ import annotations

from collections import Counter

from icframe.domain.incentive_spec import MetricScope, MetricType

from .types import CompiledMetric, RuntimeEvent


class OnlineMetrics:
    """Streaming metric reducers whose memory is independent of run length."""

    def __init__(self, metrics: tuple[CompiledMetric, ...]) -> None:
        self.metrics = metrics
        self.event_count = 0
        self.action_counts: Counter[str] = Counter()
        self.transition_counts: Counter[str] = Counter()
        self.tag_counts: Counter[str] = Counter()
        self._sums: Counter[str] = Counter()
        self._counts: Counter[str] = Counter()

    def update(self, event: RuntimeEvent) -> None:
        if event.counts_as_action:
            self.event_count += 1
            self.action_counts[event.action] += 1
            self.transition_counts[event.transition_id] += 1
        self.tag_counts.update(event.tags)
        event_tags = set(event.tags)
        for metric in self.metrics:
            name = metric.name
            if metric.type in {MetricType.SUM, MetricType.MEAN}:
                values = self._channel_values(event, metric.channel or "", metric.scope)
                self._sums[name] += sum(values)
                self._counts[name] += len(values)
            elif metric.type in {MetricType.EVENT_COUNT, MetricType.EVENT_RATE}:
                if not event.counts_as_action:
                    continue
                if not metric.required_tags or metric.required_tags.issubset(event_tags):
                    self._counts[name] += 1

    def snapshot(self) -> dict[str, float]:
        results: dict[str, float] = {}
        for metric in self.metrics:
            name = metric.name
            if metric.type is MetricType.SUM:
                results[name] = float(self._sums[name])
            elif metric.type is MetricType.MEAN:
                count = self._counts[name]
                results[name] = float(self._sums[name] / count) if count else 0.0
            elif metric.type is MetricType.EVENT_COUNT:
                results[name] = float(self._counts[name])
            elif metric.type is MetricType.EVENT_RATE:
                results[name] = (
                    float(self._counts[name] / self.event_count) if self.event_count else 0.0
                )
            elif metric.type is MetricType.DIFFERENCE:
                results[name] = results[metric.left] - results[metric.right]
            elif metric.type is MetricType.RATIO:
                denominator = results[metric.denominator]
                results[name] = results[metric.numerator] / denominator if denominator else 0.0
            elif metric.type is MetricType.WEIGHTED_SUM:
                results[name] = sum(
                    weight * results[reference] for reference, weight in metric.terms
                )
            else:  # pragma: no cover - exhaustive enum guard
                raise AssertionError(metric.type)
        return results

    @staticmethod
    def _channel_values(event: RuntimeEvent, channel: str, scope: MetricScope) -> list[float]:
        values: list[float] = []
        if scope in {MetricScope.GLOBAL, MetricScope.ALL} and channel in event.global_outcome:
            values.append(float(event.global_outcome[channel]))
        if scope in {MetricScope.AGENTS, MetricScope.ALL}:
            values.extend(
                float(outcome[channel])
                for outcome in event.outcomes_by_agent.values()
                if channel in outcome
            )
        return values
