from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class RoundCandidate:
    model_name: str
    objective: float
    score: float
    satisfied: bool
    adjacency: np.ndarray


@dataclass
class BoostRound:
    round_index: int
    chosen_model: str
    chosen_objective: float
    model_weights: Dict[str, float]
    candidates: List[RoundCandidate]


@dataclass
class ExperimentSummary:
    best_model: str
    best_objective: float
    best_score: float
    satisfied: bool
    per_model_best_objective: Dict[str, float]
    rounds: List[BoostRound]


class BoostOrchestrator(ABC):
    @abstractmethod
    def run(self) -> ExperimentSummary:
        raise NotImplementedError
