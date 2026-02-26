from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from configs.schema import ComponentConfig, ExperimentConfig, ModelConfig


def _component(data: Dict[str, Any]) -> ComponentConfig:
    return ComponentConfig(name=data["name"], params=data.get("params", {}))


def _model(data: Dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        name=data["name"],
        params=data.get("params", {}),
        enabled=bool(data.get("enabled", True)),
    )


def load_config(path: str) -> ExperimentConfig:
    payload = json.loads(Path(path).read_text())
    return ExperimentConfig(
        num_nodes=int(payload["num_nodes"]),
        rounds=int(payload.get("rounds", 10)),
        trials=int(payload.get("trials", 5)),
        seed=int(payload.get("seed", 0)),
        edge_probability=float(payload.get("edge_probability", 0.3)),
        eta=float(payload.get("eta", 1.0)),
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
        models=[_model(item) for item in payload.get("models", [])]
        or [ModelConfig(name="diffusion_search"), ModelConfig(name="energy_search")],
    )
