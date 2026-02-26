from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np


Goal = Literal["satisfy", "violate"]


@dataclass(frozen=True)
class ConjectureDiagnostics:
    score: float
    objective: float
    satisfied: bool
    invariants: Dict[str, float]


class Conjecture(ABC):
    """Base interface for graph conjectures.

    `score(adj) >= 0` means the conjecture is satisfied.
    Objective is always maximized and depends on `goal`.
    """

    def __init__(self, goal: Goal = "satisfy") -> None:
        self.goal = goal

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def score(self, adjacency: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def invariants(self, adjacency: np.ndarray) -> Dict[str, float]:
        raise NotImplementedError

    def objective(self, adjacency: np.ndarray) -> float:
        margin = float(self.score(adjacency))
        return margin if self.goal == "satisfy" else -margin

    def is_satisfied(self, adjacency: np.ndarray) -> bool:
        return self.score(adjacency) >= 0.0

    def diagnostics(self, adjacency: np.ndarray) -> ConjectureDiagnostics:
        return ConjectureDiagnostics(
            score=float(self.score(adjacency)),
            objective=float(self.objective(adjacency)),
            satisfied=bool(self.is_satisfied(adjacency)),
            invariants=self.invariants(adjacency),
        )
