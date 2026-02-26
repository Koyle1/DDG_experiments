from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from representations.base import GraphRepresentation


@dataclass
class TrainingMetrics:
    values: Dict[str, Any] = field(default_factory=dict)


class TrainableGraphGenerator(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def fit(
        self,
        elite_graphs: List[np.ndarray],
        population_graphs: List[np.ndarray],
        representation: GraphRepresentation,
        rng: np.random.Generator,
    ) -> TrainingMetrics:
        raise NotImplementedError

    @abstractmethod
    def sample_graphs(
        self,
        num_samples: int,
        num_nodes: int,
        representation: GraphRepresentation,
        rng: np.random.Generator,
    ) -> List[np.ndarray]:
        raise NotImplementedError
