from __future__ import annotations

from typing import List, Tuple

import numpy as np

from conjectures.base import Conjecture
from models.base import GraphSearchModel, SearchResult, SearchTraceStep
from representations.base import GraphRepresentation


def _edge_pairs(num_nodes: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]


class DiffusionGraphSearchModel(GraphSearchModel):
    """Lightweight diffusion-style search over adjacency matrices."""

    def __init__(
        self,
        steps: int = 120,
        beta_start: float = 0.01,
        beta_end: float = 0.20,
        guidance_scale: float = 0.6,
        guidance_edges: int = 64,
        edge_temperature: float = 0.2,
        local_refiner=None,
    ) -> None:
        super().__init__(local_refiner=local_refiner)
        self.steps = int(steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.guidance_scale = float(guidance_scale)
        self.guidance_edges = int(guidance_edges)
        self.edge_temperature = float(edge_temperature)

    @property
    def name(self) -> str:
        return "diffusion_search"

    def _search(
        self,
        adjacency: np.ndarray,
        conjecture: Conjecture,
        representation: GraphRepresentation,
        rng: np.random.Generator,
    ) -> SearchResult:
        current = representation.validate(adjacency)
        x = current.copy().astype(float)
        best = current
        best_obj = conjecture.objective(best)
        trace: List[SearchTraceStep] = []
        pairs = _edge_pairs(current.shape[0])

        betas = np.linspace(self.beta_start, self.beta_end, self.steps)
        for step, beta in enumerate(betas):
            noise = rng.normal(0.0, 1.0, size=x.shape)
            noise = np.triu(noise, k=1)
            noise = noise + noise.T
            x = (1.0 - beta) * x + beta * 0.5 + np.sqrt(beta) * noise

            # Score-guidance proxy: estimate edge-wise objective deltas.
            bin_graph = representation.validate((x >= 0.5).astype(float))
            guidance = np.zeros_like(x)
            if pairs:
                k = min(len(pairs), self.guidance_edges)
                idx = rng.choice(len(pairs), size=k, replace=False)
                base_obj = conjecture.objective(bin_graph)
                for pair_idx in idx:
                    i, j = pairs[int(pair_idx)]
                    candidate = bin_graph.copy()
                    candidate[i, j] = 1.0 - candidate[i, j]
                    candidate[j, i] = candidate[i, j]
                    delta = conjecture.objective(candidate) - base_obj
                    guidance[i, j] = delta
                    guidance[j, i] = delta

            x = x + self.guidance_scale * guidance
            x = np.clip(x, 0.0, 1.0)

            logits = (x - 0.5) / max(1e-9, self.edge_temperature)
            probs = 1.0 / (1.0 + np.exp(-logits))
            sampled = rng.binomial(1, probs).astype(float)
            sampled = representation.validate(sampled)

            obj = conjecture.objective(sampled)
            accepted = obj >= best_obj
            if accepted:
                best = sampled
                best_obj = obj

            trace.append(
                SearchTraceStep(
                    step=step,
                    objective=float(best_obj),
                    score=float(conjecture.score(best)),
                    accepted=accepted,
                    metadata={"beta": float(beta)},
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
