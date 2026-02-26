from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ComponentConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratorConfig(ComponentConfig):
    decode_per_generation: int = 16
    enabled: bool = True


@dataclass
class DiffusionBoostConfig:
    num_nodes: int
    generations: int = 20
    seed: int = 0
    edge_probability: float = 0.3
    database_size: int = 128
    elite_fraction: float = 0.25
    init_pool_factor: int = 4
    sample_regime: Optional[Dict[str, Any]] = None
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
        default_factory=lambda: ComponentConfig(
            name="greedy_edge_flip",
            params={"max_steps": 25, "candidate_edges_per_step": 80},
        )
    )
    generators: List[GeneratorConfig] = field(
        default_factory=lambda: [
            GeneratorConfig(
                name="trainable_diffusion",
                decode_per_generation=20,
            ),
            GeneratorConfig(
                name="trainable_energy",
                decode_per_generation=20,
            ),
        ]
    )
