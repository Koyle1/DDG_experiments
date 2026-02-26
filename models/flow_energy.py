# hybrid_flow_energy_circle_packing.py
# Hybrid: Flow Matching (CFM) on centers + Energy guidance + SRP/L-BFGS/LP push loop

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import minimize, linprog


# ============================
# Config
# ============================

@dataclass
class CirclePackingConfig:
    n: int
    device: str = "mps"
    dtype: torch.dtype = torch.float32

    # Penalty weights for SRP surrogate
    w_wall: float = 50.0
    w_ovlp: float = 50.0

    # SRP refinement
    srp_steps: int = 200
    srp_step_size: float = 1e-2
    srp_noise_std: float = 2e-3

    # ODE sampling
    ode_steps: int = 40
    ode_dt: float = 1.0 / 40.0

    # Energy guidance
    guidance_lambda: float = 0.05

    # Training
    lr_flow: float = 2e-4
    lr_energy: float = 2e-4


# ============================
# Constraints + surrogate loss
# ============================

def wall_violation(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    """Containment: r_i <= x_i <= 1-r_i and same for y."""
    x = centers[..., 0]
    y = centers[..., 1]
    v = (
        F.relu(radii - x) + F.relu(x + radii - 1.0) +
        F.relu(radii - y) + F.relu(y + radii - 1.0)
    )
    return v  # (n,)


def overlap_violation(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    # non-overlap: ||c_i - c_j|| >= r_i + r_j
    diffs = centers[:, None, :] - centers[None, :, :]
    d = torch.sqrt((diffs**2).sum(-1) + 1e-12)
    rr = radii[:, None] + radii[None, :]
    v = F.relu(rr - d)

    # zero diagonal WITHOUT in-place ops
    n = v.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=v.device)
    v = v * mask  # broadcasted; safe for autograd
    return v


def srp_surrogate_loss(centers: torch.Tensor, radii: torch.Tensor, cfg: CirclePackingConfig) -> torch.Tensor:
    """Penalty surrogate: wall + overlap - sum(r)."""
    wall = wall_violation(centers, radii).pow(2).sum()
    ovlp = overlap_violation(centers, radii).pow(2).sum() * 0.5  # pairs counted twice
    obj = -radii.sum()
    return cfg.w_wall * wall + cfg.w_ovlp * ovlp + obj


def reward_sum_radii(radii: np.ndarray) -> float:
    return float(np.sum(radii))


# ============================
# LP: best radii for fixed centers
# ============================

def lp_best_radii(centers: np.ndarray) -> np.ndarray:
    """
    Maximize sum r_i subject to:
      - containment: r_i <= x_i, 1-x_i, y_i, 1-y_i
      - non-overlap: r_i + r_j <= ||c_i - c_j||
      - r_i >= 0
    """
    n = centers.shape[0]
    A_ub = []
    b_ub = []

    x = centers[:, 0]
    y = centers[:, 1]

    for i in range(n):
        row = np.zeros(n); row[i] = 1.0
        A_ub.append(row); b_ub.append(x[i])
        A_ub.append(row); b_ub.append(1.0 - x[i])
        A_ub.append(row); b_ub.append(y[i])
        A_ub.append(row); b_ub.append(1.0 - y[i])

    for i in range(n):
        for j in range(i + 1, n):
            dij = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n); row[i] = 1.0; row[j] = 1.0
            A_ub.append(row); b_ub.append(dij)

    A_ub = np.stack(A_ub, axis=0)
    b_ub = np.array(b_ub)

    # maximize sum r  <=>  minimize -sum r
    c = -np.ones(n)
    bounds = [(0.0, None)] * n

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return np.full(n, 1e-4, dtype=np.float64)

    r = np.maximum(0.0, res.x - 1e-10)  # tiny safety shrink
    return r


# ============================
# Push refinement: SRP + optional L-BFGS-B on centers + LP radii
# ============================

def push_refine(centers_init: np.ndarray, cfg: CirclePackingConfig) -> Tuple[np.ndarray, np.ndarray]:
    n = cfg.n

    # Initialize radii from LP (often much better than a constant)
    r0 = lp_best_radii(centers_init)

    centers = torch.tensor(centers_init, device=cfg.device, dtype=cfg.dtype, requires_grad=True)
    radii = torch.tensor(r0, device=cfg.device, dtype=cfg.dtype, requires_grad=True)

    opt = torch.optim.SGD([centers, radii], lr=cfg.srp_step_size)

    for _ in range(cfg.srp_steps):
        # stochastic relaxation perturbation
        with torch.no_grad():
            centers.add_(cfg.srp_noise_std * torch.randn_like(centers))
            radii.add_(cfg.srp_noise_std * torch.randn_like(radii))
            centers.clamp_(0.0, 1.0)
            radii.clamp_(min=0.0)

        opt.zero_grad(set_to_none=True)
        loss = srp_surrogate_loss(centers, radii, cfg)
        loss.backward()

        # normalized gradient step for stability
        with torch.no_grad():
            for p in [centers, radii]:
                g = p.grad
                p.grad = g / (torch.norm(g) + 1e-12)

        opt.step()

        with torch.no_grad():
            centers.clamp_(0.0, 1.0)
            radii.clamp_(min=0.0)

    centers_np = centers.detach().cpu().numpy().astype(np.float64)

    # Optional polish: optimize centers using true objective through LP radii
    def scipy_obj(c_flat: np.ndarray) -> float:
        c = c_flat.reshape(n, 2)
        r = lp_best_radii(c)
        return -reward_sum_radii(r)

    bounds = [(0.0, 1.0)] * (2 * n)
    res = minimize(scipy_obj, centers_np.reshape(-1), method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
    centers_np = res.x.reshape(n, 2)

    radii_np = lp_best_radii(centers_np)
    return centers_np, radii_np


# ============================
# Models: Flow velocity + Energy
# ============================

class DeepSetsEncoder(nn.Module):
    def __init__(self, d_in=2, d_h=128, d_out=128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_h), nn.SiLU(),
            nn.Linear(d_h, d_h), nn.SiLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(d_h, d_h), nn.SiLU(),
            nn.Linear(d_h, d_out), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,2)
        h = self.phi(x)      # (B,N,H)
        s = h.mean(dim=1)    # (B,H)
        return self.rho(s)   # (B,Out)


class FlowVelocity(nn.Module):
    def __init__(self, d_h=128):
        super().__init__()
        self.enc = DeepSetsEncoder(2, d_h, d_h)
        self.mlp = nn.Sequential(
            nn.Linear(d_h + 1, d_h), nn.SiLU(),
            nn.Linear(d_h, d_h), nn.SiLU(),
            nn.Linear(d_h, 2),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B,N,2), t: (B,)
        B, N, _ = x.shape
        g = self.enc(x)              # (B,H)
        z = torch.cat([g, t[:, None]], dim=-1)
        v = self.mlp(z)              # (B,2)
        return v[:, None, :].expand(B, N, 2)


class EnergyModel(nn.Module):
    def __init__(self, d_h=128):
        super().__init__()
        self.enc = DeepSetsEncoder(2, d_h, d_h)
        self.head = nn.Sequential(
            nn.Linear(d_h, d_h), nn.SiLU(),
            nn.Linear(d_h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x))  # (B,1)


# ============================
# CFM training loss
# ============================

def cfm_loss(flow: FlowVelocity, x1: torch.Tensor) -> torch.Tensor:
    """
    Conditional Flow Matching (simple variant):
      x0 ~ prior, t ~ U(0,1), xt = (1-t)x0 + tx1
      target velocity: x1 - x0
      regress v_theta(xt,t) to target.
    """
    B, N, _ = x1.shape
    x0 = torch.rand_like(x1)
    t = torch.rand(B, device=x1.device)
    xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    v_tgt = x1 - x0
    v_pred = flow(xt, t)
    return F.mse_loss(v_pred, v_tgt)


# ============================
# Hybrid sampling: flow + energy guidance
# ============================

@torch.no_grad()
def sample_centers(flow: FlowVelocity, energy: EnergyModel, cfg: CirclePackingConfig, batch: int) -> torch.Tensor:
    x = torch.rand(batch, cfg.n, 2, device=cfg.device, dtype=cfg.dtype)  # prior in [0,1]^2

    for k in range(cfg.ode_steps):
        t = torch.full((batch,), (k + 0.5) / cfg.ode_steps, device=cfg.device, dtype=cfg.dtype)

        # enable grads for energy guidance
        x.requires_grad_(True)
        v = flow(x, t)

        E = energy(x).sum()
        (g,) = torch.autograd.grad(E, x, create_graph=False)

        with torch.no_grad():
            x = x + cfg.ode_dt * v - cfg.ode_dt * cfg.guidance_lambda * g
            x.clamp_(0.0, 1.0)

    return x.detach()


# ============================
# One boosting round training
# ============================

def train_round(
    flow: FlowVelocity,
    energy: EnergyModel,
    elite_centers: np.ndarray,   # (M,N,2)
    elite_rewards: np.ndarray,   # (M,)
    cfg: CirclePackingConfig,
    steps_flow: int = 2000,
    steps_energy: int = 2000,
    batch: int = 64,
) -> None:
    x = torch.tensor(elite_centers, device=cfg.device, dtype=cfg.dtype)
    y = torch.tensor(-elite_rewards, device=cfg.device, dtype=cfg.dtype)[:, None]  # energy target

    opt_f = torch.optim.Adam(flow.parameters(), lr=cfg.lr_flow)
    opt_e = torch.optim.Adam(energy.parameters(), lr=cfg.lr_energy)

    # Flow training
    for _ in range(steps_flow):
        idx = torch.randint(0, x.shape[0], (batch,), device=cfg.device)
        loss = cfm_loss(flow, x[idx])
        opt_f.zero_grad(set_to_none=True)
        loss.backward()
        opt_f.step()

    # Energy training
    for _ in range(steps_energy):
        idx = torch.randint(0, x.shape[0], (batch,), device=cfg.device)
        pred = energy(x[idx])
        loss = F.mse_loss(pred, y[idx])
        opt_e.zero_grad(set_to_none=True)
        loss.backward()
        opt_e.step()


# ============================
# Outer loop
# ============================

def run(cfg: CirclePackingConfig, rounds: int = 3, init_pop: int = 1000, sample_pop: int = 256) -> None:
    flow = FlowVelocity().to(cfg.device)
    energy = EnergyModel().to(cfg.device)

    # Initial pool: random -> push
    centers_pool = []
    rewards_pool = []

    for i in range(init_pop):
        c0 = np.random.rand(cfg.n, 2)
        c1, r1 = push_refine(c0, cfg)

        if i % 20 == 0:
            print(f"init sample {i}/{init_pop}")
            
        centers_pool.append(c1)
        rewards_pool.append(reward_sum_radii(r1))

    centers_pool = np.stack(centers_pool)
    rewards_pool = np.array(rewards_pool)

    for rd in range(rounds):
        # Keep top quartile as elite
        thr = np.quantile(rewards_pool, 0.75)
        elite_mask = rewards_pool >= thr
        elite_centers = centers_pool[elite_mask]
        elite_rewards = rewards_pool[elite_mask]

        train_round(flow, energy, elite_centers, elite_rewards, cfg)

        # Sample new centers using hybrid sampler
        x_new = sample_centers(flow, energy, cfg, batch=sample_pop).cpu().numpy()

        # Push refine sampled centers
        new_centers = []
        new_rewards = []
        for i in range(sample_pop):
            c1, r1 = push_refine(x_new[i], cfg)
            new_centers.append(c1)
            new_rewards.append(reward_sum_radii(r1))

        centers_pool = np.concatenate([centers_pool, np.stack(new_centers)], axis=0)
        rewards_pool = np.concatenate([rewards_pool, np.array(new_rewards)], axis=0)

        print(f"[round {rd}] best sum radii so far = {rewards_pool.max():.6f}")


if __name__ == "__main__":
    cfg = CirclePackingConfig(n=10, guidance_lambda=0.05)
    run(cfg, rounds=2, init_pop=500, sample_pop=128)
