from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from boost.pattern_boost import PatternBoostExperiment
from configs.loader import load_config
from conjectures.registry import create_conjecture
from models.registry import create_model, create_refiner
from representations.registry import create_representation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PatternBoost graph-search experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/example_experiment.json",
        help="Path to experiment config JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    representation = create_representation(
        cfg.representation.name, cfg.representation.params
    )
    conjecture = create_conjecture(cfg.conjecture.name, cfg.conjecture.params)
    refiner = (
        create_refiner(cfg.refiner.name, cfg.refiner.params)
        if cfg.refiner is not None
        else None
    )
    models = [
        create_model(m.name, m.params, local_refiner=refiner)
        for m in cfg.models
        if m.enabled
    ]

    experiment = PatternBoostExperiment(
        models=models,
        representation=representation,
        conjecture=conjecture,
        num_nodes=cfg.num_nodes,
        rounds=cfg.rounds,
        edge_probability=cfg.edge_probability,
        seed=cfg.seed,
        eta=cfg.eta,
    )
    benchmark = experiment.benchmark(trials=cfg.trials)
    summary = experiment.run()

    output = {
        "config_path": args.config,
        "benchmark": [asdict(item) for item in benchmark],
        "summary": {
            "best_model": summary.best_model,
            "best_objective": summary.best_objective,
            "best_score": summary.best_score,
            "satisfied": summary.satisfied,
            "per_model_best_objective": summary.per_model_best_objective,
            "rounds": [
                {
                    "round_index": item.round_index,
                    "chosen_model": item.chosen_model,
                    "chosen_objective": item.chosen_objective,
                    "model_weights": item.model_weights,
                    "candidates": [
                        {
                            "model_name": cand.model_name,
                            "objective": cand.objective,
                            "score": cand.score,
                            "satisfied": cand.satisfied,
                        }
                        for cand in item.candidates
                    ],
                }
                for item in summary.rounds
            ],
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
