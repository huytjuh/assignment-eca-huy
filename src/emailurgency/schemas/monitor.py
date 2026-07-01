from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict


class LexiconPrototype(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    prototype: str
    source: str
    similarity: float

    sentence: str
    keybert_score: float

    @property
    def to_csv(self) -> dict[str, str | float]:
        return {"prototype": self.prototype, "source": self.source, "similarity": self.similarity}


class LabelSignal(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    name: str
    vote: int
    weight: int
    matches: tuple[str, ...] = ()


@dataclass(frozen=True)
class SilverLabel:
    name: str
    vote: int
    weight: int
    matches: tuple[str, ...] = ()
    signals: tuple[LabelSignal, ...] = field(default_factory=tuple)

    @property
    def coverage(self) -> dict[str, int | bool]:
        return {
            "covered": self.vote != -1,
            "n_signals": len(self.signals),
            "n_matches": len(self.matches),
        }

    @property
    def to_dict(self) -> dict[str, Any]:
        return {
            "weak_label": self.vote,
            "weak_score": self.weight,
            "matched_prototypes": list(self.matches),
            "label_functions": [signal.name for signal in self.signals],
            **self.coverage,
        }

    @staticmethod
    def coverage_summary(labels: list[SilverLabel]) -> dict[str, float | int]:
        total = len(labels)
        covered = sum(label.coverage["covered"] for label in labels)
        return {
            "n": total,
            "covered": covered,
            "coverage": covered / total if total else 0.0,
        }
