from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from .base import ICFrameModel


class NormLayer(StrEnum):
    HARD = "hard"
    MORAL = "moral"
    INCENTIVE = "incentive"


class LawProgram(ICFrameModel):
    description: str = ""
    facts: list[str] = Field(default_factory=list)
    hard_rules: list[str] = Field(default_factory=list)
    moral_rules: list[str] = Field(default_factory=list)
    incentive_rules: list[str] = Field(default_factory=list)
    query_rules: list[str] = Field(default_factory=list)
    show: tuple[str, ...] = ("allowed/2", "forbidden/2", "violation/2")

    def render(self) -> str:
        sections = [
            "% facts",
            *self.facts,
            "% hard rules",
            *self.hard_rules,
            "% moral rules",
            *self.moral_rules,
            "% incentive rules",
            *self.incentive_rules,
            "% query rules",
            *self.query_rules,
            *(f"#defined {item}." for item in self.show),
            *(f"#show {item}." for item in self.show),
        ]
        return "\n".join(line for line in sections if line)


class LawEvaluation(ICFrameModel):
    allowed: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    forbidden: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    violations: dict[str, tuple[str, ...]] = Field(default_factory=dict)
