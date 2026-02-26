from __future__ import annotations

from typing import List, Tuple

import numpy as np

from conjectures.base import Conjecture
from models.base import GraphSearchModel, SearchResult, SearchTraceStep
from representations.base import GraphRepresentation


def _edge_pairs(num_nodes: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]


class EnergyBasedGraphSearchModel(GraphSearchModel):
    """Metropolis search over graph energy induced by conjecture objective."""

    def __init__(
        self,
        steps: int = 2000,
        temp_start: float = 1.0,
        temp_end: float = 0.05,
        random_restart_prob: float = 0.01,
        local_refiner=None,
    ) -> None:
        super().__init__(local_refiner=local_refiner)
        self.steps = int(steps)
        self.temp_start = float(temp_start)
        self.temp_end = float(temp_end)
        self.random_restart_prob = float(random_restart_prob)

    @property
    def name(self) -> str:
        return "energy_search"

    def _search(
        self,
        adjacency: np.ndarray,
        conjecture: Conjecture,
        representation: GraphRepresentation,
        rng: np.random.Generator,
    ) -> SearchResult:
        current = representation.validate(adjacency)
        current_obj = conjecture.objective(current)
        best = current
        best_obj = current_obj
        trace: List[SearchTraceStep] = []
        pairs = _edge_pairs(current.shape[0])

        temperatures = np.linspace(self.temp_start, self.temp_end, self.steps)
        for step, temp in enumerate(temperatures):
            if not pairs:
                break

            if rng.uniform() < self.random_restart_prob:
                density = np.triu(current, k=1).mean() * 2.0
                current = representation.sample_initial(
                    num_nodes=current.shape[0],
                    rng=rng,
                    edge_probability=float(np.clip(density, 0.01, 0.99)),
                )
                current_obj = conjecture.objective(current)

            pair_idx = int(rng.integers(0, len(pairs)))
            i, j = pairs[pair_idx]
            candidate = current.copy()
            candidate[i, j] = 1.0 - candidate[i, j]
            candidate[j, i] = candidate[i, j]

            candidate_obj = conjecture.objective(candidate)
            delta = candidate_obj - current_obj
            accept = delta >= 0.0
            if not accept:
                accept_prob = np.exp(delta / max(temp, 1e-12))
                accept = bool(rng.uniform() < accept_prob)

            if accept:
                current = candidate
                current_obj = candidate_obj

            if current_obj >= best_obj:
                best = current
                best_obj = current_obj

            trace.append(
                SearchTraceStep(
                    step=step,
                    objective=float(best_obj),
                    score=float(conjecture.score(best)),
                    accepted=accept,
                    metadata={"temp": float(temp), "delta": float(delta)},
                )
            )

        return SearchResult(
            model_name=self.name,
            adjacency=best,
            objective=float(conjecture.objective(best)),
            score=float(conjecture.score(best)),
            satisfied=bool(conjecture.is_satisfied(best)),
            trace=trace,
        )
