from __future__ import annotations

from typing import Any, Callable, Dict

from representations.adjacency_matrix import AdjacencyMatrixRepresentation
from representations.base import GraphRepresentation


RepresentationFactory = Callable[..., GraphRepresentation]


REPRESENTATION_REGISTRY: Dict[str, RepresentationFactory] = {
    "adjacency_matrix": AdjacencyMatrixRepresentation,
}


def create_representation(
    name: str, params: Dict[str, Any]
) -> GraphRepresentation:
    try:
        factory = REPRESENTATION_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(REPRESENTATION_REGISTRY))
        raise ValueError(
            f"Unknown representation '{name}'. Available: {available}"
        ) from exc
    return factory(**params)
