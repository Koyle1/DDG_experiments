# 20-graph-embedding-gp

Grammar-guided genetic programming for geometry-preserving graph embeddings.

This experiment follows the structure of `experiments-alogos` experiment 14:
- island model GP with migration,
- expression simplification and complexity penalty,
- constant fitting with L-BFGS-B restarts + Nelder-Mead fallback,
- Pareto tracking by complexity.

## Objective

Learn a symbolic 2D embedding function:

`x = f(node_features, neighbor_features), y = g(node_features, neighbor_features)`

and optimize:

`loss = stress_weight * pairwise_stress + knn_weight * (1 - kNN_overlap) + complexity_penalty`

Where:
- `pairwise_stress` compares graph geodesic distances vs embedding Euclidean distances.
- `kNN_overlap` compares nearest neighbors in graph metric vs embedding metric.

## Files

- `learn_embedding_gp.py`: main training loop.
- `island_gp.py`: multi-island GP wrapper and migration.
- `evaluation.py`: objective, metrics, constant optimization, caches.
- `data.py`: synthetic graph dataset generation and preprocessing.
- `grammars.py`: grammar definitions.
- `simplification.py`: SymPy simplification + complexity counting.
- `primitives.py`: safe numeric helpers.
- `run.sh`: default run command.

## Quick start

```bash
cd /Users/koyle/project/DDG_experiments/20-graph-embedding-gp
./run.sh
```

Or run directly:

```bash
/Users/koyle/project/DDG_experiments/venv/bin/python3 learn_embedding_gp.py --generations 80 --num-islands 6
```
