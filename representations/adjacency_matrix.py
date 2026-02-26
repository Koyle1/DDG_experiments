from __future__ import annotations

import numpy as np

from representations.base import GraphRepresentation


class AdjacencyMatrixRepresentation(GraphRepresentation):
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)

    @property
    def name(self) -> str:
        return "adjacency_matrix"

    def encode(self, adjacency: np.ndarray) -> np.ndarray:
        return self.validate(adjacency)

    def decode(self, representation: np.ndarray) -> np.ndarray:
        arr = np.asarray(representation, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("Adjacency matrix must be square.")
        binarized = (arr >= self.threshold).astype(float)
        return self.validate(binarized)

    def sample_initial(
        self,
        num_nodes: int,
        rng: np.random.Generator,
        edge_probability: float,
    ) -> np.ndarray:
        upper = rng.binomial(
            n=1, p=edge_probability, size=(num_nodes, num_nodes)
        ).astype(float)
        upper = np.triu(upper, k=1)
        adjacency = upper + upper.T
        return self.validate(adjacency)

    def validate(self, adjacency: np.ndarray) -> np.ndarray:
        arr = np.asarray(adjacency, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("Adjacency matrix must be square.")
        arr = np.where(arr > 0.5, 1.0, 0.0)
        arr = np.triu(arr, k=1)
        arr = arr + arr.T
        np.fill_diagonal(arr, 0.0)
        return arr
