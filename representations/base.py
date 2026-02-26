from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class GraphRepresentation(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def encode(self, adjacency: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def decode(self, representation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def sample_initial(
        self,
        num_nodes: int,
        rng: np.random.Generator,
        edge_probability: float,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def validate(self, adjacency: np.ndarray) -> np.ndarray:
        raise NotImplementedError
