from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from models.generative.gnn_transformer import EnergyGNNTransformer
from models.generative.vector_utils import (
    adjacency_to_edge_vector,
    edge_vector_to_adjacency,
)
from models.trainable_base import TrainableGraphGenerator, TrainingMetrics
from representations.base import GraphRepresentation


@dataclass
class TrainableEnergyConfig:
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    train_epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    sampling_steps: int = 400
    temp_start: float = 1.0
    temp_end: float = 0.05
    device: str = "cpu"


class TrainableEnergyGenerator(TrainableGraphGenerator):
    def __init__(self, **kwargs) -> None:
        self.cfg = TrainableEnergyConfig(**kwargs)
        self._num_nodes: int | None = None
        self._num_edges: int | None = None
        self._net: EnergyGNNTransformer | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._device = torch.device(self.cfg.device)

    @property
    def name(self) -> str:
        return "trainable_energy"

    def _ensure_initialized(self, num_nodes: int) -> None:
        num_edges = num_nodes * (num_nodes - 1) // 2
        if self._net is not None and self._num_nodes == num_nodes:
            return
        self._num_nodes = num_nodes
        self._num_edges = num_edges
        self._net = EnergyGNNTransformer(
            num_nodes=num_nodes,
            hidden_dim=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            num_heads=self.cfg.num_heads,
            dropout=self.cfg.dropout,
        ).to(self._device)
        self._optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

    def fit(
        self,
        elite_graphs: List[np.ndarray],
        population_graphs: List[np.ndarray],
        representation: GraphRepresentation,
        rng: np.random.Generator,
    ) -> TrainingMetrics:
        if not elite_graphs:
            return TrainingMetrics(values={"loss": float("nan")})
        num_nodes = elite_graphs[0].shape[0]
        self._ensure_initialized(num_nodes)
        assert self._net is not None
        assert self._optimizer is not None
        assert self._num_edges is not None

        seed = int(rng.integers(0, 2**31 - 1))
        torch.manual_seed(seed)

        pos = np.stack([adjacency_to_edge_vector(g) for g in elite_graphs], axis=0)
        if population_graphs:
            neg_pool = np.stack(
                [adjacency_to_edge_vector(g) for g in population_graphs], axis=0
            )
        else:
            neg_pool = rng.binomial(
                1, 0.5, size=(max(1, len(pos)), self._num_edges)
            ).astype(np.float32)

        pos_t = torch.tensor(pos, dtype=torch.float32, device=self._device)
        neg_pool_t = torch.tensor(neg_pool, dtype=torch.float32, device=self._device)
        n_pos = pos_t.shape[0]
        n_neg = neg_pool_t.shape[0]

        self._net.train()
        losses: List[float] = []
        for _ in range(self.cfg.train_epochs):
            batch_size = min(self.cfg.batch_size, n_pos)
            pos_idx = torch.randint(0, n_pos, (batch_size,), device=self._device)
            neg_idx = torch.randint(0, n_neg, (batch_size,), device=self._device)
            pos_batch = pos_t[pos_idx]
            neg_batch = neg_pool_t[neg_idx]

            e_pos = self._net(pos_batch)
            e_neg = self._net(neg_batch)
            # Contrastive ranking loss: elite graphs should have lower energy.
            loss = torch.nn.functional.softplus(e_pos - e_neg).mean()

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        return TrainingMetrics(values={"loss": float(np.mean(losses))})

    def sample_graphs(
        self,
        num_samples: int,
        num_nodes: int,
        representation: GraphRepresentation,
        rng: np.random.Generator,
    ) -> List[np.ndarray]:
        self._ensure_initialized(num_nodes)
        assert self._net is not None
        assert self._num_edges is not None

        seed = int(rng.integers(0, 2**31 - 1))
        torch.manual_seed(seed)

        self._net.eval()
        samples: List[np.ndarray] = []
        edges = np.arange(self._num_edges, dtype=np.int64)
        temperatures = np.linspace(
            self.cfg.temp_start, self.cfg.temp_end, self.cfg.sampling_steps
        )

        for _ in range(num_samples):
            state = rng.binomial(1, 0.5, size=self._num_edges).astype(np.float32)
            for temp in temperatures:
                flip_idx = int(rng.choice(edges))
                candidate = state.copy()
                candidate[flip_idx] = 1.0 - candidate[flip_idx]

                with torch.no_grad():
                    e_cur = float(
                        self._net(
                            torch.tensor(
                                state[None, :], dtype=torch.float32, device=self._device
                            )
                        )
                        .cpu()
                        .item()
                    )
                    e_cand = float(
                        self._net(
                            torch.tensor(
                                candidate[None, :],
                                dtype=torch.float32,
                                device=self._device,
                            )
                        )
                        .cpu()
                        .item()
                    )
                delta = e_cand - e_cur
                accept = delta <= 0.0
                if not accept:
                    accept = rng.uniform() < np.exp(-delta / max(temp, 1e-12))
                if accept:
                    state = candidate

            adjacency = edge_vector_to_adjacency(state, num_nodes=num_nodes)
            samples.append(representation.validate(adjacency))

        return samples
