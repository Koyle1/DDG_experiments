from __future__ import annotations

from typing import Dict, Literal

import numpy as np

from conjectures.base import Conjecture


FormanStatistic = Literal[
    "mean_forman_curvature",
    "min_forman_curvature",
    "max_forman_curvature",
    "std_forman_curvature",
    "positive_forman_fraction",
]
Relation = Literal["ge", "le"]


def _forman_edge_curvatures(adjacency: np.ndarray) -> np.ndarray:
    """Forman-Ricci edge curvature for simple unweighted graphs.

    For edge (u, v): F(u, v) = 4 - deg(u) - deg(v)
    """

    n = adjacency.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=float)
    degrees = adjacency.sum(axis=1)
    rows, cols = np.triu_indices(n, k=1)
    edge_mask = adjacency[rows, cols] > 0.5
    if not np.any(edge_mask):
        return np.zeros((0,), dtype=float)
    u = rows[edge_mask]
    v = cols[edge_mask]
    return (4.0 - degrees[u] - degrees[v]).astype(float)


class FormanCurvatureConjecture(Conjecture):
    """Threshold conjecture on Forman-Ricci curvature statistics."""

    SUPPORTED_STATS = (
        "mean_forman_curvature",
        "min_forman_curvature",
        "max_forman_curvature",
        "std_forman_curvature",
        "positive_forman_fraction",
    )

    def __init__(
        self,
        statistic: FormanStatistic = "mean_forman_curvature",
        threshold: float = 0.0,
        relation: Relation = "ge",
        goal: str = "satisfy",
    ) -> None:
        super().__init__(goal=goal)
        if statistic not in self.SUPPORTED_STATS:
            raise ValueError(
                f"Unknown statistic '{statistic}'. Supported: {self.SUPPORTED_STATS}"
            )
        if relation not in ("ge", "le"):
            raise ValueError("relation must be one of: 'ge', 'le'")
        self.statistic = statistic
        self.threshold = float(threshold)
        self.relation = relation

    @property
    def name(self) -> str:
        return "forman_curvature"

    def invariants(self, adjacency: np.ndarray) -> Dict[str, float]:
        curvatures = _forman_edge_curvatures(adjacency)
        edge_count = int(curvatures.shape[0])
        if edge_count == 0:
            mean_c = 0.0
            min_c = 0.0
            max_c = 0.0
            std_c = 0.0
            positive_fraction = 0.0
        else:
            mean_c = float(np.mean(curvatures))
            min_c = float(np.min(curvatures))
            max_c = float(np.max(curvatures))
            std_c = float(np.std(curvatures))
            positive_fraction = float(np.mean(curvatures > 0.0))

        return {
            "edge_count": float(edge_count),
            "mean_forman_curvature": mean_c,
            "min_forman_curvature": min_c,
            "max_forman_curvature": max_c,
            "std_forman_curvature": std_c,
            "positive_forman_fraction": positive_fraction,
        }

    def score(self, adjacency: np.ndarray) -> float:
        inv = self.invariants(adjacency)
        value = float(inv[self.statistic])
        if self.relation == "ge":
            return value - self.threshold
        return self.threshold - value
