from __future__ import annotations

import argparse
from copy import deepcopy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from boost.diffusion_boost import DiffusionBoostTrainer, GeneratorSpec
from boost.pattern_boost import PatternBoostExperiment
from conjectures.registry import create_conjecture
from models.registry import create_model, create_refiner
from models.trainable_registry import create_trainable_generator
from representations.registry import create_representation

# Configure training runs here.
RUNS: Dict[str, Dict[str, Any]] = {
    "diffusion_boost": {
        "enabled": True,
        "verbose": False,
        "save_to": "outputs/diffusion_boost_result.json",
        "config": {
            "num_nodes": 14,
            "generations": 10,
            "seed": 11,
            "edge_probability": 0.35,
            "database_size": 10000,
            "elite_fraction": 0.25,
            "init_pool_factor": 4,
            "sample_regime": {
                "name": "paper_like",
                "starter_samples": 1000,
                "total_decode_per_generation": 1000
            },
            "representation": {"name": "adjacency_matrix", "params": {"threshold": 0.5}},
            "conjecture": {
                "name": "linear_invariant",
                "params": {
                    "goal": "violate",
                    "weights": {"max_degree": 1.0, "m": -0.2, "triangle_count": 0.15},
                    "bias": -2.0,
                },
            },
            "refiner": {
                "name": "greedy_edge_flip",
                "params": {"max_steps": 8, "candidate_edges_per_step": 24},
            },
            "generators": [
                {
                    "enabled": True,
                    "name": "trainable_diffusion",
                    "decode_per_generation": 20,
                    "params": {
                        "timesteps": 48,
                        "hidden_dim": 256,
                        "num_layers": 3,
                        "num_heads": 8,
                        "dropout": 0.1,
                        "train_epochs": 30,
                        "batch_size": 24,
                        "learning_rate": 0.001,
                        "sample_temperature": 0.9,
                        "device": "auto",
                    },
                },
                {
                    "enabled": True,
                    "name": "trainable_energy",
                    "decode_per_generation": 20,
                    "params": {
                        "hidden_dim": 256,
                        "num_layers": 3,
                        "num_heads": 8,
                        "dropout": 0.1,
                        "train_epochs": 35,
                        "batch_size": 24,
                        "learning_rate": 0.001,
                        "sampling_steps": 350,
                        "temp_start": 1.0,
                        "temp_end": 0.05,
                        "device": "auto",
                    },
                },
            ],
        },
    },
    "pattern_boost": {
        "enabled": False,
        "verbose": False,
        "save_to": "outputs/pattern_boost_result.json",
        "config": {
            "num_nodes": 14,
            "rounds": 8,
            "trials": 4,
            "seed": 7,
            "edge_probability": 0.35,
            "eta": 0.8,
            "representation": {"name": "adjacency_matrix", "params": {"threshold": 0.5}},
            "conjecture": {
                "name": "linear_invariant",
                "params": {
                    "goal": "violate",
                    "weights": {"max_degree": 1.0, "m": -0.2, "triangle_count": 0.15},
                    "bias": -2.0,
                },
            },
            "refiner": {
                "name": "greedy_edge_flip",
                "params": {"max_steps": 25, "candidate_edges_per_step": 80},
            },
            "models": [
                {
                    "enabled": True,
                    "name": "diffusion_search",
                    "params": {
                        "steps": 80,
                        "beta_start": 0.01,
                        "beta_end": 0.25,
                        "guidance_scale": 0.5,
                        "guidance_edges": 70,
                        "edge_temperature": 0.18,
                    },
                },
                {
                    "enabled": True,
                    "name": "energy_search",
                    "params": {
                        "steps": 1600,
                        "temp_start": 1.2,
                        "temp_end": 0.03,
                        "random_restart_prob": 0.015,
                    },
                },
            ],
        },
    },
    "forman_curvature_boost": {
        "enabled": False,
        "verbose": False,
        "save_to": "outputs/forman_curvature_boost_result.json",
        "config": {
            "num_nodes": 14,
            "generations": 10,
            "seed": 11,
            "edge_probability": 0.35,
            "database_size": 10000,
            "elite_fraction": 0.25,
            "init_pool_factor": 4,
            "sample_regime": {
                "name": "paper_like",
                "starter_samples": 1000,
                "total_decode_per_generation": 1000
            },
            "representation": {"name": "adjacency_matrix", "params": {"threshold": 0.5}},
            "conjecture": {
                "name": "forman_curvature",
                "params": {
                    "goal": "violate",
                    "statistic": "mean_forman_curvature",
                    "relation": "ge",
                    "threshold": -1.0
                },
            },
            "refiner": {
                "name": "greedy_edge_flip",
                "params": {"max_steps": 8, "candidate_edges_per_step": 24},
            },
            "generators": [
                {
                    "enabled": True,
                    "name": "trainable_diffusion",
                    "decode_per_generation": 20,
                    "params": {
                        "timesteps": 48,
                        "hidden_dim": 256,
                        "num_layers": 3,
                        "num_heads": 8,
                        "dropout": 0.1,
                        "train_epochs": 30,
                        "batch_size": 24,
                        "learning_rate": 0.001,
                        "sample_temperature": 0.9,
                        "device": "auto",
                    },
                },
                {
                    "enabled": True,
                    "name": "trainable_energy",
                    "decode_per_generation": 20,
                    "params": {
                        "hidden_dim": 256,
                        "num_layers": 3,
                        "num_heads": 8,
                        "dropout": 0.1,
                        "train_epochs": 35,
                        "batch_size": 24,
                        "learning_rate": 0.001,
                        "sampling_steps": 350,
                        "temp_start": 1.0,
                        "temp_end": 0.05,
                        "device": "auto",
                    },
                },
            ],
        },
    },
}


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _dump_json(path: str, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    Path(path).write_text(json.dumps(payload, indent=2))


def _split_even(total: int, k: int) -> List[int]:
    if k <= 0:
        return []
    base = total // k
    rem = total % k
    return [base + (1 if i < rem else 0) for i in range(k)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph training pipelines.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for all selected runs.",
    )
    parser.add_argument(
        "--run",
        choices=["all", "diffusion_boost", "pattern_boost", "forman_curvature_boost"],
        default="all",
        help="Choose which run to execute.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help=(
            "Override both starter_samples and total_decode_per_generation "
            "for diffusion-style runs."
        ),
    )
    parser.add_argument(
        "--starter-samples",
        type=int,
        default=None,
        help="Override starter_samples for diffusion-style runs.",
    )
    parser.add_argument(
        "--decode-samples",
        type=int,
        default=None,
        help="Override total_decode_per_generation for diffusion-style runs.",
    )
    return parser.parse_args()


def run_diffusion_boost(cfg: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    representation = create_representation(
        cfg["representation"]["name"], cfg["representation"].get("params", {})
    )
    conjecture = create_conjecture(
        cfg["conjecture"]["name"], cfg["conjecture"].get("params", {})
    )
    refiner_cfg = cfg.get("refiner")
    refiner = (
        create_refiner(refiner_cfg["name"], refiner_cfg.get("params", {}))
        if refiner_cfg is not None
        else None
    )

    enabled_generators = [
        item for item in cfg.get("generators", []) if item.get("enabled", True)
    ]
    sample_regime = cfg.get("sample_regime")
    regime_name = None
    starter_samples = None
    decode_counts: List[int] = []
    if sample_regime is not None:
        regime_name = sample_regime.get("name", "custom")
        if "starter_samples" in sample_regime:
            starter_samples = int(sample_regime["starter_samples"])
        if "total_decode_per_generation" in sample_regime:
            total_decode = int(sample_regime["total_decode_per_generation"])
            decode_counts = _split_even(total_decode, len(enabled_generators))

    generator_specs: List[GeneratorSpec] = []
    for idx, item in enumerate(enabled_generators):
        decode_per_generation = (
            decode_counts[idx]
            if decode_counts
            else int(item.get("decode_per_generation", 16))
        )
        generator_specs.append(
            GeneratorSpec(
                generator=create_trainable_generator(
                    item["name"], item.get("params", {})
                ),
                decode_per_generation=decode_per_generation,
            )
        )

    trainer = DiffusionBoostTrainer(
        generator_specs=generator_specs,
        representation=representation,
        conjecture=conjecture,
        num_nodes=int(cfg["num_nodes"]),
        database_size=int(cfg.get("database_size", 128)),
        init_samples=starter_samples,
        elite_fraction=float(cfg.get("elite_fraction", 0.25)),
        generations=int(cfg.get("generations", 20)),
        init_pool_factor=int(cfg.get("init_pool_factor", 4)),
        edge_probability=float(cfg.get("edge_probability", 0.3)),
        local_refiner=refiner,
        seed=int(cfg.get("seed", 0)),
        verbose=verbose,
    )
    summary = trainer.run()
    return {
        "type": "diffusion_boost",
        "sample_regime": regime_name,
        "starter_samples": starter_samples,
        "decode_per_generation_total": int(
            np.sum([spec.decode_per_generation for spec in generator_specs])
        ),
        "decode_per_generation_by_model": {
            spec.generator.name: spec.decode_per_generation for spec in generator_specs
        },
        "best_objective": summary.best_objective,
        "best_score": summary.best_score,
        "satisfied": summary.satisfied,
        "best_source": summary.best_source,
        "final_database_size": summary.final_database_size,
        "generations": [
            {
                "generation": item.generation,
                "database_best_objective": item.database_best_objective,
                "database_avg_objective": item.database_avg_objective,
                "model_stats": [asdict(stat) for stat in item.model_stats],
            }
            for item in summary.generations
        ],
    }


def run_pattern_boost(cfg: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    representation = create_representation(
        cfg["representation"]["name"], cfg["representation"].get("params", {})
    )
    conjecture = create_conjecture(
        cfg["conjecture"]["name"], cfg["conjecture"].get("params", {})
    )
    refiner_cfg = cfg.get("refiner")
    refiner = (
        create_refiner(refiner_cfg["name"], refiner_cfg.get("params", {}))
        if refiner_cfg is not None
        else None
    )
    models = [
        create_model(item["name"], item.get("params", {}), local_refiner=refiner)
        for item in cfg.get("models", [])
        if item.get("enabled", True)
    ]
    experiment = PatternBoostExperiment(
        models=models,
        representation=representation,
        conjecture=conjecture,
        num_nodes=int(cfg["num_nodes"]),
        rounds=int(cfg.get("rounds", 10)),
        edge_probability=float(cfg.get("edge_probability", 0.3)),
        seed=int(cfg.get("seed", 0)),
        eta=float(cfg.get("eta", 1.0)),
        verbose=verbose,
    )
    benchmark = experiment.benchmark(trials=int(cfg.get("trials", 5)))
    summary = experiment.run()
    return {
        "type": "pattern_boost",
        "benchmark": [asdict(item) for item in benchmark],
        "summary": {
            "best_model": summary.best_model,
            "best_objective": summary.best_objective,
            "best_score": summary.best_score,
            "satisfied": summary.satisfied,
            "per_model_best_objective": summary.per_model_best_objective,
        },
    }


def main() -> None:
    args = parse_args()
    all_results: Dict[str, Any] = {}
    for run_name, run_spec in RUNS.items():
        if args.run != "all" and run_name != args.run:
            continue
        if args.run == "all" and not run_spec.get("enabled", False):
            continue
        cfg = deepcopy(run_spec["config"])
        if run_name in ("diffusion_boost", "forman_curvature_boost"):
            sample_regime = dict(cfg.get("sample_regime", {}))
            if args.samples is not None:
                sample_regime["starter_samples"] = int(args.samples)
                sample_regime["total_decode_per_generation"] = int(args.samples)
            if args.starter_samples is not None:
                sample_regime["starter_samples"] = int(args.starter_samples)
            if args.decode_samples is not None:
                sample_regime["total_decode_per_generation"] = int(args.decode_samples)
            if sample_regime:
                cfg["sample_regime"] = sample_regime
        verbose = bool(run_spec.get("verbose", False) or args.verbose)
        if verbose:
            print(f"[train] starting run={run_name}")
        if run_name in ("diffusion_boost", "forman_curvature_boost"):
            result = run_diffusion_boost(cfg, verbose=verbose)
        elif run_name == "pattern_boost":
            result = run_pattern_boost(cfg, verbose=verbose)
        else:
            raise ValueError(f"Unknown run '{run_name}' in RUNS config.")

        save_to = run_spec.get("save_to")
        if save_to:
            _dump_json(save_to, result)
            if verbose:
                print(f"[train] wrote {save_to}")
        all_results[run_name] = result

    if not all_results:
        print("{}")
        return
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
