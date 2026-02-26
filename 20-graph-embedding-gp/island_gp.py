"""Island model GP wrapper for alogos."""

from __future__ import annotations

import copy
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import alogos as al
import numpy as np


@dataclass
class IslandState:
    island_id: int
    ea: al.EvolutionaryAlgorithm
    best_fitness: float = float("inf")
    best_phenotype: str = ""
    generation: int = 0
    gens_without_improvement: int = 0


@dataclass
class MigrationEvent:
    generation: int
    from_island: int
    to_island: int
    phenotype: str
    fitness: float


@dataclass
class ParetoEntry:
    phenotype: str
    fitness: float
    complexity: int
    generation: int


class ParetoHallOfFame:
    def __init__(self, objective: str = "min"):
        self.objective = objective
        self.front: Dict[int, ParetoEntry] = {}

    def _is_better(self, new_fitness: float, old_fitness: float) -> bool:
        if self.objective == "min":
            return new_fitness < old_fitness
        return new_fitness > old_fitness

    def update(self, phenotype: str, fitness: float, complexity: int, generation: int) -> bool:
        if not np.isfinite(fitness):
            return False
        if complexity not in self.front:
            self.front[complexity] = ParetoEntry(phenotype, fitness, complexity, generation)
            return True
        if self._is_better(fitness, self.front[complexity].fitness):
            self.front[complexity] = ParetoEntry(phenotype, fitness, complexity, generation)
            return True
        return False

    def get_pareto_front(self) -> List[ParetoEntry]:
        if not self.front:
            return []
        entries = sorted(self.front.values(), key=lambda e: e.complexity)
        pareto: List[ParetoEntry] = []
        best_fitness = float("inf") if self.objective == "min" else float("-inf")
        for entry in entries:
            if self.objective == "min":
                if entry.fitness < best_fitness:
                    pareto.append(entry)
                    best_fitness = entry.fitness
            else:
                if entry.fitness > best_fitness:
                    pareto.append(entry)
                    best_fitness = entry.fitness
        return pareto


class IslandModelGP:
    """Grammar-guided GP with multiple islands and migration."""

    def __init__(
        self,
        grammar: al.Grammar,
        objective_function: Callable,
        objective: str = "min",
        num_islands: int = 5,
        migration_interval: int = 25,
        migration_size: int = 2,
        topology: str = "ring",
        population_size: int = 20,
        offspring_size: int = 100,
        system: str = "cfggpst",
        verbose: int = 0,
        complexity_fn: Optional[Callable[[str], int]] = None,
        parallel: bool = True,
        **kwargs,
    ):
        self.grammar = grammar
        self.objective_function = objective_function
        self.objective = objective
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.topology = topology
        self.verbose = verbose
        self.complexity_fn = complexity_fn or (lambda x: len(x))
        self.parallel = parallel and num_islands > 1

        self.ea_kwargs = {
            "system": system,
            "population_size": population_size,
            "offspring_size": offspring_size,
            "verbose": 0,
            **kwargs,
        }

        self.islands: List[IslandState] = []
        for i in range(num_islands):
            ea = al.EvolutionaryAlgorithm(grammar, objective_function, objective, **self.ea_kwargs)
            self.islands.append(IslandState(island_id=i, ea=ea))

        self.generation = 0
        self.best_fitness = float("inf") if objective == "min" else float("-inf")
        self.best_phenotype = ""
        self.best_island = -1
        self.migration_history: List[MigrationEvent] = []
        self.hall_of_fame = ParetoHallOfFame(objective)

    def _is_better(self, new_fitness: float, old_fitness: float) -> bool:
        if self.objective == "min":
            return new_fitness < old_fitness
        return new_fitness > old_fitness

    def _get_individuals(self, island: IslandState):
        pop = island.ea.state.population
        if pop is None:
            return []
        if hasattr(pop, "individuals"):
            return pop.individuals
        return list(pop)

    def _set_individuals(self, island: IslandState, individuals):
        pop = island.ea.state.population
        if pop is not None and hasattr(pop, "individuals"):
            pop.individuals = individuals

    @staticmethod
    def _step_island(island: IslandState):
        return island.ea.step()

    def _update_island(self, island: IslandState, best):
        old_best = island.best_fitness
        island.generation = island.ea.state.generation
        island.best_fitness = best.fitness
        island.best_phenotype = best.phenotype
        if self._is_better(best.fitness, old_best):
            island.gens_without_improvement = 0
        else:
            island.gens_without_improvement += 1

        complexity = self.complexity_fn(best.phenotype)
        self.hall_of_fame.update(best.phenotype, best.fitness, complexity, self.generation)

        if self._is_better(best.fitness, self.best_fitness):
            self.best_fitness = best.fitness
            self.best_phenotype = best.phenotype
            self.best_island = island.island_id

    def step(self):
        self.generation += 1
        if self.parallel:
            with ThreadPoolExecutor(max_workers=self.num_islands) as pool:
                futures = [pool.submit(self._step_island, isl) for isl in self.islands]
                results = [f.result() for f in futures]
            for island, best in zip(self.islands, results):
                self._update_island(island, best)
        else:
            for island in self.islands:
                best = self._step_island(island)
                self._update_island(island, best)

        if self.generation % self.migration_interval == 0:
            self._migrate()

        return self.best_phenotype, self.best_fitness

    def _migrate(self):
        if self.topology == "ring":
            self._migrate_ring()
        elif self.topology == "fully_connected":
            self._migrate_fully_connected()
        elif self.topology == "random":
            self._migrate_random()
        else:
            raise ValueError(f"Unknown topology: {self.topology}")

    def _migrate_ring(self):
        migrants = []
        for island in self.islands:
            individuals = self._get_individuals(island)
            if individuals:
                ranked = sorted(individuals, key=lambda ind: ind.fitness, reverse=(self.objective == "max"))
                migrants.append(ranked[: self.migration_size])
            else:
                migrants.append([])

        for i, island in enumerate(self.islands):
            src = (i - 1) % self.num_islands
            recv = migrants[src]
            individuals = self._get_individuals(island)
            if recv and individuals:
                ranked = sorted(individuals, key=lambda ind: ind.fitness, reverse=(self.objective == "min"))
                for j, migrant in enumerate(recv):
                    if j < len(ranked):
                        ranked[j] = copy.deepcopy(migrant)
                        self.migration_history.append(
                            MigrationEvent(
                                generation=self.generation,
                                from_island=src,
                                to_island=i,
                                phenotype=migrant.phenotype,
                                fitness=migrant.fitness,
                            )
                        )
                self._set_individuals(island, ranked)

    def _migrate_fully_connected(self):
        best_each = []
        for island in self.islands:
            individuals = self._get_individuals(island)
            if individuals:
                if self.objective == "min":
                    best = min(individuals, key=lambda x: x.fitness)
                else:
                    best = max(individuals, key=lambda x: x.fitness)
                best_each.append((island.island_id, best))
        if not best_each:
            return
        if self.objective == "min":
            src_id, global_best = min(best_each, key=lambda x: x[1].fitness)
        else:
            src_id, global_best = max(best_each, key=lambda x: x[1].fitness)

        for island in self.islands:
            if island.island_id == src_id:
                continue
            individuals = self._get_individuals(island)
            if not individuals:
                continue
            if self.objective == "min":
                worst_idx = max(range(len(individuals)), key=lambda i: individuals[i].fitness)
            else:
                worst_idx = min(range(len(individuals)), key=lambda i: individuals[i].fitness)
            individuals[worst_idx] = copy.deepcopy(global_best)
            self.migration_history.append(
                MigrationEvent(
                    generation=self.generation,
                    from_island=src_id,
                    to_island=island.island_id,
                    phenotype=global_best.phenotype,
                    fitness=global_best.fitness,
                )
            )

    def _migrate_random(self):
        migrants = []
        for island in self.islands:
            individuals = self._get_individuals(island)
            if individuals:
                ranked = sorted(individuals, key=lambda ind: ind.fitness, reverse=(self.objective == "max"))
                migrants.append(ranked[: self.migration_size])
            else:
                migrants.append([])

        for i, island in enumerate(self.islands):
            candidates = [j for j in range(self.num_islands) if j != i]
            individuals = self._get_individuals(island)
            if not candidates or not individuals:
                continue
            src = random.choice(candidates)
            recv = migrants[src]
            if not recv:
                continue
            ranked = sorted(individuals, key=lambda ind: ind.fitness, reverse=(self.objective == "min"))
            for j, migrant in enumerate(recv):
                if j < len(ranked):
                    ranked[j] = copy.deepcopy(migrant)
                    self.migration_history.append(
                        MigrationEvent(
                            generation=self.generation,
                            from_island=src,
                            to_island=i,
                            phenotype=migrant.phenotype,
                            fitness=migrant.fitness,
                        )
                    )
            self._set_individuals(island, ranked)

    def get_pareto_front(self) -> List[ParetoEntry]:
        return self.hall_of_fame.get_pareto_front()
