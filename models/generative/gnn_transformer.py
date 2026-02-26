from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn


class _GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.message_proj = nn.Linear(hidden_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_states: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # Message passing uses the current weighted graph as neighborhood structure.
        deg = adjacency.sum(dim=-1, keepdim=True).clamp(min=1.0)
        neigh = torch.bmm(adjacency, self.message_proj(node_states)) / deg
        node_states = self.norm1(node_states + self.dropout(neigh))

        attn_out, _ = self.self_attn(node_states, node_states, node_states, need_weights=False)
        node_states = self.norm2(node_states + self.dropout(attn_out))

        ffn_out = self.ffn(node_states)
        node_states = self.norm3(node_states + self.dropout(ffn_out))
        return node_states


class GNNTransformerBackbone(nn.Module):
    """Graph encoder combining message passing and transformer attention."""

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        time_conditioned: bool,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.time_conditioned = bool(time_conditioned)

        rows, cols = np.triu_indices(self.num_nodes, k=1)
        self.register_buffer("_rows", torch.tensor(rows, dtype=torch.long))
        self.register_buffer("_cols", torch.tensor(cols, dtype=torch.long))

        self.node_emb = nn.Embedding(self.num_nodes, self.hidden_dim)
        self.degree_proj = nn.Linear(1, self.hidden_dim)
        self.time_proj = (
            nn.Sequential(
                nn.Linear(1, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            if self.time_conditioned
            else None
        )
        self.layers = nn.ModuleList(
            _GraphTransformerLayer(
                hidden_dim=self.hidden_dim, num_heads=num_heads, dropout=dropout
            )
            for _ in range(num_layers)
        )

    @property
    def num_edges(self) -> int:
        return int(self._rows.numel())

    def _adjacency_from_edge_vector(self, edge_vector: torch.Tensor) -> torch.Tensor:
        batch_size = edge_vector.shape[0]
        adjacency = edge_vector.new_zeros((batch_size, self.num_nodes, self.num_nodes))
        adjacency[:, self._rows, self._cols] = edge_vector
        adjacency[:, self._cols, self._rows] = edge_vector
        return adjacency

    def encode_nodes(
        self, edge_vector: torch.Tensor, t_norm: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        adjacency = self._adjacency_from_edge_vector(edge_vector)
        degrees = adjacency.sum(dim=-1, keepdim=True) / max(1, self.num_nodes - 1)

        node_ids = torch.arange(self.num_nodes, device=edge_vector.device)
        node_states = self.node_emb(node_ids).unsqueeze(0).expand(edge_vector.shape[0], -1, -1)
        node_states = node_states + self.degree_proj(degrees)

        if self.time_conditioned:
            if t_norm is None:
                raise ValueError("t_norm is required for time-conditioned backbone.")
            node_states = node_states + self.time_proj(t_norm).unsqueeze(1)

        for layer in self.layers:
            node_states = layer(node_states, adjacency)
        return node_states, adjacency

    def edge_features(
        self, node_states: torch.Tensor, edge_vector: torch.Tensor, t_norm: torch.Tensor | None = None
    ) -> torch.Tensor:
        hi = node_states[:, self._rows, :]
        hj = node_states[:, self._cols, :]
        edge_scalar = edge_vector.unsqueeze(-1)

        pieces = [hi, hj, torch.abs(hi - hj), edge_scalar]
        if self.time_conditioned:
            if t_norm is None:
                raise ValueError("t_norm is required for time-conditioned backbone.")
            t_rep = t_norm.unsqueeze(1).expand(-1, self.num_edges, -1)
            pieces.append(t_rep)
        return torch.cat(pieces, dim=-1)


class DiffusionGNNTransformer(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = GNNTransformerBackbone(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            time_conditioned=True,
        )
        edge_feat_dim = 3 * hidden_dim + 2
        self.noise_head = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_t: torch.Tensor, t_norm: torch.Tensor) -> torch.Tensor:
        x_graph = torch.sigmoid(x_t)
        node_states, _ = self.backbone.encode_nodes(x_graph, t_norm=t_norm)
        edge_feat = self.backbone.edge_features(node_states, x_graph, t_norm=t_norm)
        pred_noise = self.noise_head(edge_feat).squeeze(-1)
        return pred_noise


class EnergyGNNTransformer(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = GNNTransformerBackbone(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            time_conditioned=False,
        )
        edge_feat_dim = 3 * hidden_dim + 1
        self.edge_energy_head = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.graph_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_vector: torch.Tensor) -> torch.Tensor:
        node_states, _ = self.backbone.encode_nodes(edge_vector)
        edge_feat = self.backbone.edge_features(node_states, edge_vector)
        edge_energy = self.edge_energy_head(edge_feat).squeeze(-1).mean(dim=1)
        graph_energy = self.graph_readout(node_states.mean(dim=1)).squeeze(-1)
        return edge_energy + graph_energy
