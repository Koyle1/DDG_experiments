from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ComponentConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig(ComponentConfig):
    enabled: bool = True


@dataclass
class ExperimentConfig:
    num_nodes: int
    rounds: int = 10
    trials: int = 5
    seed: int = 0
    edge_probability: float = 0.3
    eta: float = 1.0
    representation: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(name="adjacency_matrix")
    )
    conjecture: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(
            name="linear_invariant",
            params={"weights": {"m": 1.0, "max_degree": -1.0}, "bias": 0.0},
        )
    )
    refiner: Optional[ComponentConfig] = field(
        default_factory=lambda: ComponentConfig(name="greedy_edge_flip")
    )
    models: List[ModelConfig] = field(
        default_factory=lambda: [
            ModelConfig(name="diffusion_search"),
            ModelConfig(name="energy_search"),
        ]
    )
