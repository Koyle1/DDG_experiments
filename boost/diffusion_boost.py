from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from conjectures.base import Conjecture
from models.base import LocalRefiner
from models.generative.vector_utils import adjacency_to_edge_vector
from models.trainable_base import TrainableGraphGenerator
from representations.base import GraphRepresentation


@dataclass
class GeneratorSpec:
    generator: TrainableGraphGenerator
    decode_per_generation: int


@dataclass
class DatabaseEntry:
    adjacency: np.ndarray
    objective: float
    score: float
    satisfied: bool
    source: str


@dataclass
class ModelGenerationStats:
    model_name: str
    train_metrics: Dict[str, float]
    best_objective: float
    avg_objective: float
    num_candidates: int
    num_refinement_improved: int


@dataclass
class DiffusionBoostGeneration:
    generation: int
    database_best_objective: float
    database_avg_objective: float
    model_stats: List[ModelGenerationStats] = field(default_factory=list)


@dataclass
class DiffusionBoostSummary:
    best_objective: float
    best_score: float
    satisfied: bool
    best_source: str
    generations: List[DiffusionBoostGeneration]
    final_database_size: int


class DiffusionBoostTrainer:
    """Paper-style local-global training loop with elite retraining."""

    def __init__(
        self,
        generator_specs: List[GeneratorSpec],
        representation: GraphRepresentation,
        conjecture: Conjecture,
        num_nodes: int,
        database_size: int = 128,
        init_samples: int | None = None,
        elite_fraction: float = 0.25,
        generations: int = 20,
        init_pool_factor: int = 4,
        edge_probability: float = 0.3,
        local_refiner: LocalRefiner | None = None,
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        if not generator_specs:
            raise ValueError("At least one trainable generator is required.")
        self.generator_specs = generator_specs
        self.representation = representation
        self.conjecture = conjecture
        self.num_nodes = int(num_nodes)
        self.database_size = int(database_size)
        self.init_samples = int(init_samples) if init_samples is not None else None
        self.elite_fraction = float(elite_fraction)
        self.generations = int(generations)
        self.init_pool_factor = int(init_pool_factor)
        self.edge_probability = float(edge_probability)
        self.local_refiner = local_refiner
        self.seed = int(seed)
        self.verbose = bool(verbose)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _evaluate(self, adjacency: np.ndarray, source: str, rng: np.random.Generator) -> DatabaseEntry:
        clean = self.representation.validate(adjacency)
        pre_obj = self.conjecture.objective(clean)
        improved = False
        if self.local_refiner is not None:
            refined = self.local_refiner.refine(
                clean, self.conjecture, self.representation, rng
            )
            if refined.objective >= pre_obj:
                clean = refined.adjacency
                improved = refined.objective > pre_obj
        return DatabaseEntry(
            adjacency=clean,
            objective=float(self.conjecture.objective(clean)),
            score=float(self.conjecture.score(clean)),
            satisfied=bool(self.conjecture.is_satisfied(clean)),
            source=source + ("+refine" if improved else ""),
        )

    def _dedup_top(self, entries: List[DatabaseEntry]) -> List[DatabaseEntry]:
        by_key: Dict[bytes, DatabaseEntry] = {}
        for entry in entries:
            key = adjacency_to_edge_vector(entry.adjacency).tobytes()
            old = by_key.get(key)
            if old is None or entry.objective > old.objective:
                by_key[key] = entry
        unique = list(by_key.values())
        unique.sort(key=lambda x: x.objective, reverse=True)
        return unique[: self.database_size]

    def _initialize_database(self, rng: np.random.Generator) -> List[DatabaseEntry]:
        entries: List[DatabaseEntry] = []
        target = (
            self.init_samples
            if self.init_samples is not None
            else self.database_size * self.init_pool_factor
        )
        self._log(f"[diffusion_boost] starter local-search runs={target}")
        for _ in range(target):
            base = self.representation.sample_initial(
                self.num_nodes, rng, self.edge_probability
            )
            entry = self._evaluate(base, source="init", rng=rng)
            entries.append(entry)
        return self._dedup_top(entries)

    def run(self) -> DiffusionBoostSummary:
        rng = np.random.default_rng(self.seed)
        database = self._initialize_database(rng)
        history: List[DiffusionBoostGeneration] = []
        self._log(
            f"[diffusion_boost] init database size={len(database)} "
            f"best_objective={database[0].objective:.6f}"
        )

        for generation in range(self.generations):
            elite_count = max(1, int(np.ceil(len(database) * self.elite_fraction)))
            elite = database[:elite_count]
            elite_graphs = [entry.adjacency for entry in elite]
            population_graphs = [entry.adjacency for entry in database]

            new_entries: List[DatabaseEntry] = []
            model_stats: List[ModelGenerationStats] = []
            for spec in self.generator_specs:
                train_metrics = spec.generator.fit(
                    elite_graphs=elite_graphs,
                    population_graphs=population_graphs,
                    representation=self.representation,
                    rng=rng,
                )

                raw_samples = spec.generator.sample_graphs(
                    num_samples=spec.decode_per_generation,
                    num_nodes=self.num_nodes,
                    representation=self.representation,
                    rng=rng,
                )

                candidate_entries: List[DatabaseEntry] = []
                num_improved = 0
                for sample in raw_samples:
                    pre_obj = self.conjecture.objective(sample)
                    entry = self._evaluate(sample, source=spec.generator.name, rng=rng)
                    if entry.objective > pre_obj:
                        num_improved += 1
                    candidate_entries.append(entry)

                if candidate_entries:
                    best_objective = float(
                        np.max([entry.objective for entry in candidate_entries])
                    )
                    avg_objective = float(
                        np.mean([entry.objective for entry in candidate_entries])
                    )
                else:
                    best_objective = float("-inf")
                    avg_objective = float("-inf")

                model_stats.append(
                    ModelGenerationStats(
                        model_name=spec.generator.name,
                        train_metrics=dict(train_metrics.values),
                        best_objective=best_objective,
                        avg_objective=avg_objective,
                        num_candidates=len(candidate_entries),
                        num_refinement_improved=num_improved,
                    )
                )
                new_entries.extend(candidate_entries)

            database = self._dedup_top(database + new_entries)
            if not database:
                raise RuntimeError("Database became empty during training.")

            self._log(
                f"[diffusion_boost] generation={generation + 1}/{self.generations} "
                f"db_best={database[0].objective:.6f} "
                f"db_avg={np.mean([entry.objective for entry in database]):.6f}"
            )
            for stat in model_stats:
                loss = stat.train_metrics.get("loss", float("nan"))
                self._log(
                    f"[diffusion_boost]   model={stat.model_name} "
                    f"loss={loss:.6f} best={stat.best_objective:.6f} "
                    f"avg={stat.avg_objective:.6f} "
                    f"refined={stat.num_refinement_improved}/{stat.num_candidates}"
                )

            history.append(
                DiffusionBoostGeneration(
                    generation=generation,
                    database_best_objective=float(database[0].objective),
                    database_avg_objective=float(
                        np.mean([entry.objective for entry in database])
                    ),
                    model_stats=model_stats,
                )
            )

        best = database[0]
        return DiffusionBoostSummary(
            best_objective=float(best.objective),
            best_score=float(best.score),
            satisfied=bool(best.satisfied),
            best_source=best.source,
            generations=history,
            final_database_size=len(database),
        )
