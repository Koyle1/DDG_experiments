from __future__ import annotations

from typing import Dict

import numpy as np

from conjectures.base import Conjecture


def _triangle_count(adjacency: np.ndarray) -> float:
    # For undirected simple graphs, trace(A^3) / 6 is the number of triangles.
    a3 = adjacency @ adjacency @ adjacency
    return float(np.trace(a3) / 6.0)


class LinearInvariantConjecture(Conjecture):
    """Configurable conjecture over common graph invariants.

    score = bias + sum_i(weights[i] * invariant_i)
    score >= 0 means the conjecture is satisfied.
    """

    SUPPORTED_INVARIANTS = (
        "n",
        "m",
        "avg_degree",
        "max_degree",
        "min_degree",
        "density",
        "triangle_count",
        "spectral_radius",
    )

    def __init__(
        self,
        weights: Dict[str, float],
        bias: float = 0.0,
        goal: str = "satisfy",
    ) -> None:
        super().__init__(goal=goal)
        unknown = sorted(set(weights) - set(self.SUPPORTED_INVARIANTS))
        if unknown:
            raise ValueError(
                f"Unknown invariants in weights: {unknown}. "
                f"Supported: {self.SUPPORTED_INVARIANTS}"
            )
        self.weights = dict(weights)
        self.bias = float(bias)

    @property
    def name(self) -> str:
        return "linear_invariant"

    def invariants(self, adjacency: np.ndarray) -> Dict[str, float]:
        n = adjacency.shape[0]
        upper_sum = float(np.triu(adjacency, k=1).sum())
        degrees = adjacency.sum(axis=1)
        denom = max(1, n * (n - 1) / 2)
        spectral_radius = (
            float(np.max(np.linalg.eigvalsh(adjacency))) if n > 0 else 0.0
        )
        return {
            "n": float(n),
            "m": upper_sum,
            "avg_degree": float(degrees.mean()) if n > 0 else 0.0,
            "max_degree": float(degrees.max()) if n > 0 else 0.0,
            "min_degree": float(degrees.min()) if n > 0 else 0.0,
            "density": float(upper_sum / denom),
            "triangle_count": _triangle_count(adjacency),
            "spectral_radius": spectral_radius,
        }

    def score(self, adjacency: np.ndarray) -> float:
        inv = self.invariants(adjacency)
        total = self.bias
        for name, coeff in self.weights.items():
            total += coeff * inv[name]
        return float(total)
