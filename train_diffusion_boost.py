from __future__ import annotations

import argparse
import json

import numpy as np

from boost.diffusion_boost import DiffusionBoostTrainer, GeneratorSpec
from configs.diffusion_boost_loader import load_diffusion_boost_config
from conjectures.registry import create_conjecture
from models.registry import create_refiner
from models.trainable_registry import create_trainable_generator
from representations.registry import create_representation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train diffusion/energy generators with DiffusionBoost loop."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/diffusion_boost_example.json",
        help="Path to diffusion-boost config JSON.",
    )
    return parser.parse_args()


def _split_even(total: int, k: int) -> list[int]:
    if k <= 0:
        return []
    base = total // k
    rem = total % k
    return [base + (1 if i < rem else 0) for i in range(k)]


def main() -> None:
    args = parse_args()
    cfg = load_diffusion_boost_config(args.config)

    representation = create_representation(
        cfg.representation.name, cfg.representation.params
    )
    conjecture = create_conjecture(cfg.conjecture.name, cfg.conjecture.params)
    refiner = (
        create_refiner(cfg.refiner.name, cfg.refiner.params)
        if cfg.refiner is not None
        else None
    )

    enabled_generators = [item for item in cfg.generators if item.enabled]
    sample_regime = cfg.sample_regime
    regime_name = None
    starter_samples = None
    decode_counts: list[int] = []
    if sample_regime is not None:
        regime_name = sample_regime.get("name", "custom")
        if "starter_samples" in sample_regime:
            starter_samples = int(sample_regime["starter_samples"])
        if "total_decode_per_generation" in sample_regime:
            total_decode = int(sample_regime["total_decode_per_generation"])
            decode_counts = _split_even(total_decode, len(enabled_generators))

    generator_specs = []
    for idx, item in enumerate(enabled_generators):
        decode_per_generation = (
            decode_counts[idx] if decode_counts else item.decode_per_generation
        )
        generator_specs.append(
            GeneratorSpec(
                generator=create_trainable_generator(item.name, item.params),
                decode_per_generation=decode_per_generation,
            )
        )

    trainer = DiffusionBoostTrainer(
        generator_specs=generator_specs,
        representation=representation,
        conjecture=conjecture,
        num_nodes=cfg.num_nodes,
        database_size=cfg.database_size,
        init_samples=starter_samples,
        elite_fraction=cfg.elite_fraction,
        generations=cfg.generations,
        init_pool_factor=cfg.init_pool_factor,
        edge_probability=cfg.edge_probability,
        local_refiner=refiner,
        seed=cfg.seed,
    )
    summary = trainer.run()

    output = {
        "config_path": args.config,
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
                "model_stats": [
                    {
                        "model_name": stat.model_name,
                        "train_metrics": stat.train_metrics,
                        "best_objective": stat.best_objective,
                        "avg_objective": stat.avg_objective,
                        "num_candidates": stat.num_candidates,
                        "num_refinement_improved": stat.num_refinement_improved,
                    }
                    for stat in item.model_stats
                ],
            }
            for item in summary.generations
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
