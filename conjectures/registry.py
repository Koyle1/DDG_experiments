from __future__ import annotations

from typing import Any, Callable, Dict

from conjectures.base import Conjecture
from conjectures.forman_curvature import FormanCurvatureConjecture
from conjectures.linear_invariant import LinearInvariantConjecture


ConjectureFactory = Callable[..., Conjecture]


CONJECTURE_REGISTRY: Dict[str, ConjectureFactory] = {
    "forman_curvature": FormanCurvatureConjecture,
    "linear_invariant": LinearInvariantConjecture,
}


def create_conjecture(name: str, params: Dict[str, Any]) -> Conjecture:
    try:
        factory = CONJECTURE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(CONJECTURE_REGISTRY))
        raise ValueError(
            f"Unknown conjecture '{name}'. Available: {available}"
        ) from exc
    return factory(**params)
