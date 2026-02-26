from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from configs.diffusion_boost_schema import (
    ComponentConfig,
    DiffusionBoostConfig,
    GeneratorConfig,
)


def _component(data: Dict[str, Any]) -> ComponentConfig:
    return ComponentConfig(name=data["name"], params=data.get("params", {}))


def _generator(data: Dict[str, Any]) -> GeneratorConfig:
    return GeneratorConfig(
        name=data["name"],
        params=data.get("params", {}),
        decode_per_generation=int(data.get("decode_per_generation", 16)),
        enabled=bool(data.get("enabled", True)),
    )


def load_diffusion_boost_config(path: str) -> DiffusionBoostConfig:
    payload = json.loads(Path(path).read_text())
    return DiffusionBoostConfig(
        num_nodes=int(payload["num_nodes"]),
        generations=int(payload.get("generations", 20)),
        seed=int(payload.get("seed", 0)),
        edge_probability=float(payload.get("edge_probability", 0.3)),
        database_size=int(payload.get("database_size", 128)),
        elite_fraction=float(payload.get("elite_fraction", 0.25)),
        init_pool_factor=int(payload.get("init_pool_factor", 4)),
        sample_regime=payload.get("sample_regime"),
        representation=_component(
            payload.get("representation", {"name": "adjacency_matrix"})
        ),
        conjecture=_component(
            payload.get(
                "conjecture",
                {
                    "name": "linear_invariant",
                    "params": {"weights": {"m": 1.0, "max_degree": -1.0}},
                },
            )
        ),
        refiner=(
            _component(payload["refiner"])
            if payload.get("refiner") is not None
            else None
        ),
        generators=[_generator(item) for item in payload.get("generators", [])]
        or [
            GeneratorConfig(name="trainable_diffusion"),
            GeneratorConfig(name="trainable_energy"),
        ],
    )
