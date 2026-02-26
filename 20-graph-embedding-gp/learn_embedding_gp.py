#!/usr/bin/env python3
"""Island-model GP for geometry-preserving graph embeddings."""

import argparse
import os
import random
import sys
import time

import alogos as al
import numpy as np

from data import load_data
from evaluation import Objective, evaluate_on_graphs, rebuild_expr
from grammars import GRAMMARS, DEFAULT_GRAMMAR
from island_gp import IslandModelGP
from simplification import count_ast_nodes, simplify_embedding


class TeeLogger:
    """Duplicate stdout to a log file."""

    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._file = open(log_path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


def print_pareto(island_gp: IslandModelGP, limit: int = 10):
    pareto = island_gp.get_pareto_front()
    print(f"\n--- Pareto Hall of Fame ({len(pareto)} entries) ---")
    for entry in pareto[:limit]:
        print(
            f"  complexity={entry.complexity:3d} "
            f"fitness={entry.fitness:.6f} "
            f"gen={entry.generation} "
            f"expr={entry.phenotype}"
        )


def evaluate_on_test(
    obj: Objective,
    test_graphs,
    no_simplify: bool,
    stress_weight: float,
    knn_weight: float,
    fallback_expr: str,
    label: str = "",
):
    ranked = sorted(obj.template_cache.items(), key=lambda kv: kv[1][1])[:5]
    candidates = []
    for tpl, (params, train_loss) in ranked:
        expr = rebuild_expr(tpl, params)
        candidates.append((expr, train_loss))
    if not candidates:
        candidates = [(fallback_expr, obj(fallback_expr))]

    print(f"\n--- Test evaluation{' (' + label + ')' if label else ''} ---")
    for i, (expr, train_loss) in enumerate(candidates, start=1):
        test_loss, test_stress, test_knn = evaluate_on_graphs(
            expr,
            test_graphs,
            stress_weight=stress_weight,
            knn_weight=knn_weight,
        )
        shown = expr if no_simplify else simplify_embedding(expr)
        print(
            f"  {i}. train_loss={train_loss:.6f} "
            f"test_loss={test_loss:.6f} "
            f"stress={test_stress:.6f} "
            f"knn={test_knn:.4f}"
        )
        print(f"      {shown}")


def parse_args():
    p = argparse.ArgumentParser(description="GP for geometry-preserving graph embeddings")
    p.add_argument("--generations", type=int, default=120)
    p.add_argument("--num-islands", type=int, default=5)
    p.add_argument("--migration-interval", type=int, default=20)
    p.add_argument("--migration-size", type=int, default=2)
    p.add_argument("--topology", choices=["ring", "fully_connected", "random"], default="ring")
    p.add_argument("--population", type=int, default=20)
    p.add_argument("--offspring", type=int, default=120)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--grammar", choices=list(GRAMMARS.keys()), default=DEFAULT_GRAMMAR)
    p.add_argument("--no-simplify", action="store_true")
    p.add_argument("--length-penalty", type=float, nargs="?", const=5e-4, default=0.0, metavar="W")
    p.add_argument("--max-train-graphs", type=int, default=28, help="0 means all")
    p.add_argument("--epoch-resampling", type=int, default=0, metavar="N")
    p.add_argument("--log-interval", type=int, default=0, metavar="N")
    p.add_argument("--checkpoint-interval", type=int, default=30, metavar="N")
    p.add_argument("--log-file", type=str, default=None)

    p.add_argument("--num-graphs", type=int, default=48)
    p.add_argument("--min-nodes", type=int, default=20)
    p.add_argument("--max-nodes", type=int, default=60)
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--max-pairs", type=int, default=2500)
    p.add_argument("--knn-k", type=int, default=6)

    p.add_argument("--stress-weight", type=float, default=1.0)
    p.add_argument("--knn-weight", type=float, default=0.35)
    return p.parse_args()


def main():
    args = parse_args()
    logger = TeeLogger(args.log_file) if args.log_file else None

    try:
        if args.rng_seed is not None:
            random.seed(args.rng_seed)
            np.random.seed(args.rng_seed)

        train_graphs, test_graphs = load_data(
            num_graphs=args.num_graphs,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            test_fraction=args.test_fraction,
            rng_seed=args.rng_seed,
            max_pairs=args.max_pairs,
            knn_k=args.knn_k,
        )
        n_train_nodes = sum(len(next(iter(g.vars.values()))) for g in train_graphs)
        n_test_nodes = sum(len(next(iter(g.vars.values()))) for g in test_graphs)
        print(
            f"Train: {len(train_graphs)} graphs ({n_train_nodes} nodes), "
            f"Test: {len(test_graphs)} graphs ({n_test_nodes} nodes)"
        )
        print(f"Simplification: {'disabled' if args.no_simplify else 'enabled'}")

        max_graphs = args.max_train_graphs if args.max_train_graphs > 0 else None
        complexity_penalty = (args.length_penalty, count_ast_nodes) if args.length_penalty else None
        obj = Objective(
            train_graphs,
            stress_weight=args.stress_weight,
            knn_weight=args.knn_weight,
            complexity_penalty=complexity_penalty,
            use_simplification=not args.no_simplify,
            max_graphs=max_graphs,
            seed=args.rng_seed,
        )
        print(f"Objective: {len(obj.graphs)} training graphs (max_train_graphs={args.max_train_graphs})")
        print(f"Weights: stress={args.stress_weight}, knn={args.knn_weight}")

        grammar_text = GRAMMARS[args.grammar]
        grammar = al.Grammar(bnf_text=grammar_text)
        print(f"Grammar: {args.grammar}")

        island_gp = IslandModelGP(
            grammar,
            obj,
            objective="min",
            num_islands=args.num_islands,
            migration_interval=args.migration_interval,
            migration_size=args.migration_size,
            topology=args.topology,
            population_size=args.population,
            offspring_size=args.offspring,
            max_nodes=120,
            init_pop_unique_phenotypes=True,
            complexity_fn=count_ast_nodes,
        )

        print(
            f"\nRunning {args.num_islands} islands for {args.generations} generations; "
            f"migration every {args.migration_interval} gens ({args.topology})..."
        )
        log_interval = args.log_interval if args.log_interval > 0 else args.migration_interval
        print(f"Log interval: every {log_interval} gens")
        start_time = time.time()

        for gen in range(args.generations):
            if args.epoch_resampling > 0 and gen > 0 and gen % args.epoch_resampling == 0:
                obj.resample()
                if args.verbose:
                    print(f"[gen {gen}] Resampled objective graph subset (epoch {obj._epoch})")

            phenotype, fitness = island_gp.step()

            if args.verbose and (gen + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                gens_done = gen + 1
                rate = gens_done / elapsed if elapsed > 0 else 0.0
                if (gen + 1) % args.migration_interval == 0:
                    print(
                        f"\n[gen {island_gp.generation}] Global best: fitness={fitness:.6f} "
                        f"({rate:.2f} gen/s, {elapsed:.0f}s elapsed)"
                    )
                    for isl in island_gp.islands:
                        stag = f" (stagnant:{isl.gens_without_improvement})" if isl.gens_without_improvement > 10 else ""
                        print(
                            f"  Island {isl.island_id}: fitness={isl.best_fitness:.6f}{stag} "
                            f"expr={isl.best_phenotype}"
                        )
                else:
                    print(f"[gen {gens_done}] best={fitness:.6f} ({rate:.2f} gen/s)")

            if args.checkpoint_interval > 0 and (gen + 1) % args.checkpoint_interval == 0:
                print_pareto(island_gp)
                evaluate_on_test(
                    obj=obj,
                    test_graphs=test_graphs,
                    no_simplify=args.no_simplify,
                    stress_weight=args.stress_weight,
                    knn_weight=args.knn_weight,
                    fallback_expr=phenotype,
                    label=f"gen {gen + 1}",
                )

        elapsed = time.time() - start_time
        print(f"\nCompleted {args.generations} generations in {elapsed:.1f}s")
        print(f"Best fitness: {island_gp.best_fitness:.6f}")
        print(f"Best phenotype: {island_gp.best_phenotype}")
        if not args.no_simplify:
            print(f"Simplified: {simplify_embedding(island_gp.best_phenotype)}")

        print_pareto(island_gp)
        evaluate_on_test(
            obj=obj,
            test_graphs=test_graphs,
            no_simplify=args.no_simplify,
            stress_weight=args.stress_weight,
            knn_weight=args.knn_weight,
            fallback_expr=island_gp.best_phenotype,
            label="final",
        )
        print(f"\nTotal migrations: {len(island_gp.migration_history)}")
    finally:
        if logger:
            logger.close()


if __name__ == "__main__":
    main()
