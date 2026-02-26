from __future__ import annotations

from typing import Any, Callable, Dict

from models.base import GraphSearchModel, LocalRefiner
from models.generative.diffusion import DiffusionGraphSearchModel
from models.generative.energy import EnergyBasedGraphSearchModel
from models.refinement import GreedyEdgeFlipRefiner


ModelFactory = Callable[..., GraphSearchModel]
RefinerFactory = Callable[..., LocalRefiner]


MODEL_REGISTRY: Dict[str, ModelFactory] = {
    "diffusion_search": DiffusionGraphSearchModel,
    "energy_search": EnergyBasedGraphSearchModel,
}

REFINER_REGISTRY: Dict[str, RefinerFactory] = {
    "greedy_edge_flip": GreedyEdgeFlipRefiner,
}


def create_refiner(name: str, params: Dict[str, Any]) -> LocalRefiner:
    try:
        factory = REFINER_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(REFINER_REGISTRY))
        raise ValueError(
            f"Unknown refiner '{name}'. Available: {available}"
        ) from exc
    return factory(**params)


def create_model(
    name: str,
    params: Dict[str, Any],
    local_refiner: LocalRefiner | None = None,
) -> GraphSearchModel:
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{name}'. Available: {available}"
        ) from exc
    return factory(local_refiner=local_refiner, **params)
