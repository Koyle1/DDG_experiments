from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from conjectures.base import Conjecture
from representations.base import GraphRepresentation


@dataclass
class SearchTraceStep:
    step: int
    objective: float
    score: float
    accepted: bool
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class SearchResult:
    model_name: str
    adjacency: np.ndarray
    objective: float
    score: float
    satisfied: bool
    trace: List[SearchTraceStep]


class LocalRefiner(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def refine(
        self,
        adjacency: np.ndarray,
        conjecture: Conjecture,
        representation: GraphRepresentation,
        rng: np.random.Generator,
    ) -> SearchResult:
        raise NotImplementedError


class GraphSearchModel(ABC):
    def __init__(self, local_refiner: LocalRefiner | None = None) -> None:
        self.local_refiner = local_refiner

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def search(
        self,
        adjacency: np.ndarray,
        conjecture: Conjecture,
        representation: GraphRepresentation,
        rng: np.random.Generator,
    ) -> SearchResult:
        base_result = self._search(adjacency, conjecture, representation, rng)
        if self.local_refiner is None:
            return base_result

        refined = self.local_refiner.refine(
            base_result.adjacency, conjecture, representation, rng
        )
        if refined.objective >= base_result.objective:
            merged_trace = base_result.trace + refined.trace
            return SearchResult(
                model_name=self.name,
                adjacency=refined.adjacency,
                objective=refined.objective,
                score=refined.score,
                satisfied=refined.satisfied,
                trace=merged_trace,
            )
        return base_result

    @abstractmethod
    def _search(
        self,
        adjacency: np.ndarray,
        conjecture: Conjecture,
        representation: GraphRepresentation,
        rng: np.random.Generator,
    ) -> SearchResult:
        raise NotImplementedError
