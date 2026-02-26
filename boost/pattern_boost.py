from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from boost.base import BoostOrchestrator, BoostRound, ExperimentSummary, RoundCandidate
from conjectures.base import Conjecture
from models.base import GraphSearchModel
from representations.base import GraphRepresentation


@dataclass
class BenchmarkStats:
    model_name: str
    best_objective: float
    avg_objective: float
    best_score: float
    success_rate: float


class PatternBoostExperiment(BoostOrchestrator):
    """PatternBoost-like loop that reweights models by recent gain."""

    def __init__(
        self,
        models: List[GraphSearchModel],
        representation: GraphRepresentation,
        conjecture: Conjecture,
        num_nodes: int,
        rounds: int,
        edge_probability: float,
        seed: int = 0,
        eta: float = 1.0,
        verbose: bool = False,
    ) -> None:
        if not models:
            raise ValueError("At least one model is required.")
        self.models = models
        self.representation = representation
        self.conjecture = conjecture
        self.num_nodes = int(num_nodes)
        self.rounds = int(rounds)
        self.edge_probability = float(edge_probability)
        self.seed = int(seed)
        self.eta = float(eta)
        self.verbose = bool(verbose)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def benchmark(self, trials: int = 5) -> List[BenchmarkStats]:
        rng = np.random.default_rng(self.seed)
        outputs: List[BenchmarkStats] = []
        for model in self.models:
            objs: List[float] = []
            scores: List[float] = []
            successes = 0
            for _ in range(trials):
                start = self.representation.sample_initial(
                    self.num_nodes, rng, self.edge_probability
                )
                result = model.search(
                    adjacency=start,
                    conjecture=self.conjecture,
                    representation=self.representation,
                    rng=rng,
                )
                objs.append(result.objective)
                scores.append(result.score)
                successes += int(result.satisfied)
            outputs.append(
                BenchmarkStats(
                    model_name=model.name,
                    best_objective=float(np.max(objs)),
                    avg_objective=float(np.mean(objs)),
                    best_score=float(np.max(scores)),
                    success_rate=float(successes / max(1, trials)),
                )
            )
            self._log(
                f"[pattern_boost][benchmark] model={model.name} "
                f"best={np.max(objs):.6f} avg={np.mean(objs):.6f} "
                f"success={successes}/{max(1, trials)}"
            )
        return outputs

    def run(self) -> ExperimentSummary:
        rng = np.random.default_rng(self.seed)
        current = self.representation.sample_initial(
            self.num_nodes, rng, self.edge_probability
        )
        current_obj = self.conjecture.objective(current)
        model_weights: Dict[str, float] = {
            model.name: 1.0 / len(self.models) for model in self.models
        }
        rounds: List[BoostRound] = []
        per_model_best: Dict[str, float] = {
            model.name: float("-inf") for model in self.models
        }

        best_graph = current
        best_obj = current_obj

        for round_index in range(self.rounds):
            candidates: List[RoundCandidate] = []
            gains: Dict[str, float] = {}

            for model in self.models:
                result = model.search(
                    adjacency=current,
                    conjecture=self.conjecture,
                    representation=self.representation,
                    rng=rng,
                )
                gain = result.objective - current_obj
                gains[model.name] = gain
                per_model_best[model.name] = max(
                    per_model_best[model.name], result.objective
                )
                candidates.append(
                    RoundCandidate(
                        model_name=model.name,
                        objective=result.objective,
                        score=result.score,
                        satisfied=result.satisfied,
                        adjacency=result.adjacency,
                    )
                )

            chosen = max(candidates, key=lambda c: c.objective)
            current = chosen.adjacency
            current_obj = chosen.objective
            if chosen.objective > best_obj:
                best_obj = chosen.objective
                best_graph = chosen.adjacency

            for model in self.models:
                model_weights[model.name] *= np.exp(self.eta * gains[model.name])
            normalizer = sum(model_weights.values())
            if normalizer <= 0:
                model_weights = {
                    model.name: 1.0 / len(self.models) for model in self.models
                }
            else:
                model_weights = {
                    k: float(v / normalizer) for k, v in model_weights.items()
                }

            rounds.append(
                BoostRound(
                    round_index=round_index,
                    chosen_model=chosen.model_name,
                    chosen_objective=float(chosen.objective),
                    model_weights=dict(model_weights),
                    candidates=candidates,
                )
            )
            self._log(
                f"[pattern_boost] round={round_index + 1}/{self.rounds} "
                f"chosen={chosen.model_name} objective={chosen.objective:.6f}"
            )
            self._log(
                "[pattern_boost]   weights="
                + ", ".join(
                    f"{name}:{weight:.3f}"
                    for name, weight in sorted(model_weights.items())
                )
            )

        best_score = float(self.conjecture.score(best_graph))
        return ExperimentSummary(
            best_model=max(per_model_best, key=per_model_best.get),
            best_objective=float(best_obj),
            best_score=best_score,
            satisfied=bool(self.conjecture.is_satisfied(best_graph)),
            per_model_best_objective=per_model_best,
            rounds=rounds,
        )
