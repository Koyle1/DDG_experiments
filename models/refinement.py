from __future__ import annotations

from typing import List, Tuple

import numpy as np

from conjectures.base import Conjecture
from models.base import LocalRefiner, SearchResult, SearchTraceStep
from representations.base import GraphRepresentation


def _edge_index_pairs(num_nodes: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]


class GreedyEdgeFlipRefiner(LocalRefiner):
    """Local search over single-edge flips in adjacency-matrix space."""

    def __init__(
        self,
        max_steps: int = 20,
        candidate_edges_per_step: int = 64,
        min_improvement: float = 1e-9,
    ) -> None:
        self.max_steps = int(max_steps)
        self.candidate_edges_per_step = int(candidate_edges_per_step)
        self.min_improvement = float(min_improvement)

    @property
    def name(self) -> str:
        return "greedy_edge_flip"

    def refine(
        self,
        adjacency: np.ndarray,
        conjecture: Conjecture,
        representation: GraphRepresentation,
        rng: np.random.Generator,
    ) -> SearchResult:
        current = representation.validate(adjacency)
        current_obj = conjecture.objective(current)
        trace: List[SearchTraceStep] = []
        pairs = _edge_index_pairs(current.shape[0])

        for step in range(self.max_steps):
            if not pairs:
                break
            k = min(len(pairs), self.candidate_edges_per_step)
            idx = rng.choice(len(pairs), size=k, replace=False)
            best_delta = 0.0
            best_candidate = current

            for pair_idx in idx:
                i, j = pairs[int(pair_idx)]
                candidate = current.copy()
                candidate[i, j] = 1.0 - candidate[i, j]
                candidate[j, i] = candidate[i, j]
                cand_obj = conjecture.objective(candidate)
                delta = cand_obj - current_obj
                if delta > best_delta:
                    best_delta = delta
                    best_candidate = candidate

            accepted = best_delta > self.min_improvement
            if accepted:
                current = best_candidate
                current_obj = conjecture.objective(current)

            trace.append(
                SearchTraceStep(
                    step=step,
                    objective=float(current_obj),
                    score=float(conjecture.score(current)),
                    accepted=accepted,
                    metadata={"delta": float(best_delta)},
                )
            )
            if not accepted:
                break

        return SearchResult(
            model_name=self.name,
            adjacency=current,
            objective=float(conjecture.objective(current)),
            score=float(conjecture.score(current)),
            satisfied=bool(conjecture.is_satisfied(current)),
            trace=trace,
        )
