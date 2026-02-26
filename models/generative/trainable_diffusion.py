from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from models.device_utils import resolve_device
from models.generative.gnn_transformer import DiffusionGNNTransformer
from models.generative.vector_utils import (
    adjacency_to_edge_vector,
    edge_vector_to_adjacency,
)
from models.trainable_base import TrainableGraphGenerator, TrainingMetrics
from representations.base import GraphRepresentation


@dataclass
class TrainableDiffusionConfig:
    timesteps: int = 64
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    train_epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    sample_temperature: float = 1.0
    device: str = "auto"


class TrainableDiffusionGenerator(TrainableGraphGenerator):
    def __init__(self, **kwargs) -> None:
        self.cfg = TrainableDiffusionConfig(**kwargs)
        self._num_nodes: int | None = None
        self._num_edges: int | None = None
        self._net: DiffusionGNNTransformer | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._device_name = resolve_device(self.cfg.device)
        self._device = torch.device(self._device_name)
        self._betas: torch.Tensor | None = None
        self._alphas: torch.Tensor | None = None
        self._alpha_bars: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return "trainable_diffusion"

    def _ensure_initialized(self, num_nodes: int) -> None:
        num_edges = num_nodes * (num_nodes - 1) // 2
        if self._net is not None and self._num_nodes == num_nodes:
            return
        self._num_nodes = num_nodes
        self._num_edges = num_edges
        self._net = DiffusionGNNTransformer(
            num_nodes=num_nodes,
            hidden_dim=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            num_heads=self.cfg.num_heads,
            dropout=self.cfg.dropout,
        ).to(self._device)
        self._optimizer = torch.optim.Adam(
            self._net.parameters(), lr=self.cfg.learning_rate
        )
        betas = torch.linspace(
            self.cfg.beta_start, self.cfg.beta_end, self.cfg.timesteps, device=self._device
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self._betas = betas
        self._alphas = alphas
        self._alpha_bars = alpha_bars

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
        assert self._alpha_bars is not None

        seed = int(rng.integers(0, 2**31 - 1))
        torch.manual_seed(seed)

        x0 = np.stack([adjacency_to_edge_vector(g) for g in elite_graphs], axis=0)
        x0 = 2.0 * x0 - 1.0
        x0_t = torch.tensor(x0, dtype=torch.float32, device=self._device)

        losses: List[float] = []
        n = x0_t.shape[0]
        self._net.train()
        for _ in range(self.cfg.train_epochs):
            batch_size = min(self.cfg.batch_size, n)
            indices = torch.randint(0, n, (batch_size,), device=self._device)
            x_batch = x0_t[indices]
            t = torch.randint(
                1, self.cfg.timesteps + 1, (batch_size,), device=self._device
            )
            t_idx = t - 1
            alpha_bar = self._alpha_bars[t_idx].unsqueeze(1)
            noise = torch.randn_like(x_batch)
            x_t = torch.sqrt(alpha_bar) * x_batch + torch.sqrt(1.0 - alpha_bar) * noise
            t_norm = (t.float() / float(self.cfg.timesteps)).unsqueeze(1)
            pred_noise = self._net(x_t, t_norm)
            loss = torch.mean((pred_noise - noise) ** 2)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        return TrainingMetrics(
            values={"loss": float(np.mean(losses)), "device": self._device_name}
        )

    def sample_graphs(
        self,
        num_samples: int,
        num_nodes: int,
        representation: GraphRepresentation,
        rng: np.random.Generator,
    ) -> List[np.ndarray]:
        self._ensure_initialized(num_nodes)
        assert self._net is not None
        assert self._betas is not None
        assert self._alphas is not None
        assert self._alpha_bars is not None
        assert self._num_edges is not None

        seed = int(rng.integers(0, 2**31 - 1))
        torch.manual_seed(seed)

        self._net.eval()
        with torch.no_grad():
            x_t = torch.randn((num_samples, self._num_edges), device=self._device)
            for t in range(self.cfg.timesteps, 0, -1):
                t_norm = torch.full(
                    (num_samples, 1),
                    fill_value=float(t) / float(self.cfg.timesteps),
                    device=self._device,
                )
                eps_pred = self._net(x_t, t_norm)
                beta_t = self._betas[t - 1]
                alpha_t = self._alphas[t - 1]
                alpha_bar_t = self._alpha_bars[t - 1]
                mean = (1.0 / torch.sqrt(alpha_t)) * (
                    x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
                )
                if t > 1:
                    noise = torch.randn_like(x_t)
                    sigma = torch.sqrt(beta_t) * self.cfg.sample_temperature
                    x_t = mean + sigma * noise
                else:
                    x_t = mean

            probs = torch.sigmoid(x_t)
            vectors = (probs >= 0.5).float().cpu().numpy()

        out: List[np.ndarray] = []
        for vec in vectors:
            adjacency = edge_vector_to_adjacency(vec, num_nodes=num_nodes)
            out.append(representation.validate(adjacency))
        return out
