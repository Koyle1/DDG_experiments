from __future__ import annotations

from typing import Tuple

import numpy as np


def edge_indices(num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(num_nodes, k=1)


def adjacency_to_edge_vector(adjacency: np.ndarray) -> np.ndarray:
    n = adjacency.shape[0]
    rows, cols = edge_indices(n)
    return adjacency[rows, cols].astype(np.float32)


def edge_vector_to_adjacency(edge_vector: np.ndarray, num_nodes: int) -> np.ndarray:
    rows, cols = edge_indices(num_nodes)
    vec = np.asarray(edge_vector, dtype=np.float32)
    if vec.shape[0] != rows.shape[0]:
        raise ValueError(
            f"Edge vector length {vec.shape[0]} does not match graph size {num_nodes}."
        )
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adjacency[rows, cols] = vec
    adjacency[cols, rows] = vec
    np.fill_diagonal(adjacency, 0.0)
    return adjacency
