"""Synthetic graph dataset utilities for geometry-preserving embeddings."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


@dataclass
class GraphRecord:
    """Container for one graph and precomputed training targets."""

    name: str
    vars: Dict[str, np.ndarray]
    pair_i: np.ndarray
    pair_j: np.ndarray
    gdist: np.ndarray
    knn_graph: np.ndarray


def _ensure_connected(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph
    if nx.is_connected(graph):
        return nx.convert_node_labels_to_integers(graph)
    largest_cc = max(nx.connected_components(graph), key=len)
    sub = graph.subgraph(largest_cc).copy()
    return nx.convert_node_labels_to_integers(sub)


def _graph_cycle_chords(n: int, rng: random.Random) -> nx.Graph:
    g = nx.cycle_graph(n)
    extra = max(1, n // 4)
    for _ in range(extra):
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i != j:
            g.add_edge(i, j)
    return g


def _sample_graph(kind: str, n: int, rng: random.Random) -> nx.Graph:
    seed = rng.randrange(1_000_000_000)
    if kind == "er":
        p = min(0.35, max(0.08, 2.2 * math.log(max(n, 3)) / max(n, 3)))
        g = nx.erdos_renyi_graph(n, p, seed=seed)
    elif kind == "ba":
        m = max(1, min(4, n // 6))
        g = nx.barabasi_albert_graph(n, m, seed=seed)
    elif kind == "ws":
        k = max(2, min(8, (n // 4) * 2))
        p = 0.20
        g = nx.watts_strogatz_graph(n, k, p, seed=seed)
    elif kind == "tree":
        # networkx API changed: nx.random_tree was removed in newer versions.
        if hasattr(nx, "random_tree"):
            g = nx.random_tree(n, seed=seed)
        else:
            from networkx.generators.trees import random_labeled_tree
            g = random_labeled_tree(n, seed=seed)
    elif kind == "cycle_chords":
        g = _graph_cycle_chords(n, rng)
    else:
        raise ValueError(f"Unknown graph kind: {kind}")
    return _ensure_connected(g)


def _compute_var_map(g: nx.Graph) -> Dict[str, np.ndarray]:
    n = g.number_of_nodes()
    deg = np.array([d for _, d in g.degree()], dtype=np.float64)
    deg_norm = deg / max(1.0, float(n - 1))
    clust = np.array([nx.clustering(g, i) for i in range(n)], dtype=np.float64)
    core_map = nx.core_number(g)
    core = np.array([core_map[i] for i in range(n)], dtype=np.float64)
    core = core / max(1.0, np.max(core))
    pr_map = nx.pagerank(g, max_iter=100, tol=1e-06)
    pagerank = np.array([pr_map[i] for i in range(n)], dtype=np.float64)
    idx = np.linspace(0.0, 1.0, n, dtype=np.float64)

    feats = np.column_stack([deg_norm, clust, core, pagerank, idx])
    adj = nx.to_numpy_array(g, dtype=np.float64)
    degs = np.maximum(adj.sum(axis=1, keepdims=True), 1.0)
    neigh_mean = adj @ feats / degs

    var_map: Dict[str, np.ndarray] = {}
    for j in range(feats.shape[1]):
        var_map[f"f{j}"] = feats[:, j]
        var_map[f"m{j}"] = neigh_mean[:, j]
    return var_map


def _sample_pairs(n: int, max_pairs: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    i_idx, j_idx = np.triu_indices(n, k=1)
    total = len(i_idx)
    if max_pairs <= 0 or total <= max_pairs:
        return i_idx, j_idx
    chosen = rng.choice(total, size=max_pairs, replace=False)
    return i_idx[chosen], j_idx[chosen]


def _build_record(
    graph: nx.Graph,
    name: str,
    max_pairs: int,
    knn_k: int,
    rng: np.random.Generator,
) -> GraphRecord:
    n = graph.number_of_nodes()
    dmat = nx.floyd_warshall_numpy(graph).astype(np.float64)
    i_idx, j_idx = _sample_pairs(n, max_pairs=max_pairs, rng=rng)
    gdist = dmat[i_idx, j_idx]

    dcopy = np.array(dmat, copy=True)
    np.fill_diagonal(dcopy, np.inf)
    k = min(knn_k, max(1, n - 1))
    knn_graph = np.argsort(dcopy, axis=1)[:, :k]

    return GraphRecord(
        name=name,
        vars=_compute_var_map(graph),
        pair_i=i_idx,
        pair_j=j_idx,
        gdist=gdist,
        knn_graph=knn_graph,
    )


def load_data(
    num_graphs: int = 48,
    min_nodes: int = 20,
    max_nodes: int = 60,
    test_fraction: float = 0.25,
    rng_seed: int = 42,
    max_pairs: int = 2500,
    knn_k: int = 6,
) -> Tuple[List[GraphRecord], List[GraphRecord]]:
    """Generate a synthetic train/test set of graph embedding tasks."""
    if num_graphs < 4:
        raise ValueError("num_graphs must be at least 4")
    if min_nodes < 6:
        raise ValueError("min_nodes must be >= 6")
    if max_nodes < min_nodes:
        raise ValueError("max_nodes must be >= min_nodes")

    py_rng = random.Random(rng_seed)
    np_rng = np.random.default_rng(rng_seed)
    kinds = ["er", "ba", "ws", "tree", "cycle_chords"]

    records: List[GraphRecord] = []
    for i in range(num_graphs):
        kind = kinds[i % len(kinds)]
        n = py_rng.randint(min_nodes, max_nodes)
        g = _sample_graph(kind, n, py_rng)
        rec = _build_record(
            graph=g,
            name=f"{kind}_{i}",
            max_pairs=max_pairs,
            knn_k=knn_k,
            rng=np_rng,
        )
        records.append(rec)

    idx = np.arange(len(records))
    np_rng.shuffle(idx)
    n_test = max(1, int(round(test_fraction * len(records))))
    test_ids = set(idx[:n_test].tolist())

    train = [records[i] for i in range(len(records)) if i not in test_ids]
    test = [records[i] for i in range(len(records)) if i in test_ids]
    return train, test
