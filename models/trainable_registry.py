from __future__ import annotations

from typing import Any, Callable, Dict

from models.generative.trainable_diffusion import TrainableDiffusionGenerator
from models.generative.trainable_energy import TrainableEnergyGenerator
from models.trainable_base import TrainableGraphGenerator


TrainableGeneratorFactory = Callable[..., TrainableGraphGenerator]


TRAINABLE_GENERATOR_REGISTRY: Dict[str, TrainableGeneratorFactory] = {
    "trainable_diffusion": TrainableDiffusionGenerator,
    "trainable_energy": TrainableEnergyGenerator,
}


def create_trainable_generator(
    name: str, params: Dict[str, Any]
) -> TrainableGraphGenerator:
    try:
        factory = TRAINABLE_GENERATOR_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(TRAINABLE_GENERATOR_REGISTRY))
        raise ValueError(
            f"Unknown trainable generator '{name}'. Available: {available}"
        ) from exc
    return factory(**params)
