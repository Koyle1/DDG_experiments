"""Evaluation and objective functions for graph embedding GP."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from data import GraphRecord
from primitives import CONST_RE, _GL, safe_div, sqrtabs, softclip
from simplification import simplify_embedding


def parse_embedding_expr(phenotype: str) -> Optional[Tuple[str, str]]:
    """Parse phenotype `x=<expr>;y=<expr>`."""
    parts = [p.strip() for p in phenotype.split(";") if p.strip()]
    x_expr = None
    y_expr = None
    for part in parts:
        if part.startswith("x="):
            x_expr = part[2:].strip()
        elif part.startswith("y="):
            y_expr = part[2:].strip()
    if x_expr is None or y_expr is None:
        return None
    return x_expr, y_expr


def _eval_scalar_expr(expr: str, var_map: Dict[str, np.ndarray], C=None) -> Optional[np.ndarray]:
    n = len(next(iter(var_map.values())))
    ns = {
        **var_map,
        "tanh": np.tanh,
        "log1p": np.log1p,
        "abs": np.abs,
        "sqrtabs": sqrtabs,
        "safe_div": safe_div,
        "clip": softclip,
    }
    if C is not None:
        ns["C"] = C
    try:
        out = eval(expr, _GL, ns)
    except Exception:
        return None
    arr = np.asarray(out, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    elif arr.ndim != 1 or arr.shape[0] != n:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def evaluate_embedding_expr(
    x_expr: str,
    y_expr: str,
    var_map: Dict[str, np.ndarray],
    C=None,
) -> Optional[np.ndarray]:
    x = _eval_scalar_expr(x_expr, var_map, C=C)
    if x is None:
        return None
    y = _eval_scalar_expr(y_expr, var_map, C=C)
    if y is None:
        return None
    coords = np.column_stack([x, y])
    if not np.all(np.isfinite(coords)):
        return None
    return coords


def stress_loss(coords: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray, gdist: np.ndarray) -> float:
    dx = coords[pair_i, 0] - coords[pair_j, 0]
    dy = coords[pair_i, 1] - coords[pair_j, 1]
    edist = np.sqrt(dx * dx + dy * dy + 1e-12)
    rel = (edist - gdist) / (gdist + 1e-6)
    return float(np.mean(rel * rel))


def knn_overlap(coords: np.ndarray, graph_knn: np.ndarray) -> float:
    n = coords.shape[0]
    k = min(graph_knn.shape[1], max(1, n - 1))
    if k <= 0:
        return 1.0

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
    np.fill_diagonal(dist, np.inf)
    emb_knn = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]

    overlaps = []
    for i in range(n):
        gset = set(graph_knn[i, :k].tolist())
        eset = set(emb_knn[i, :k].tolist())
        overlaps.append(len(gset.intersection(eset)) / float(k))
    return float(np.mean(overlaps))


def evaluate_on_graphs(
    phenotype: str,
    graphs: List[GraphRecord],
    stress_weight: float,
    knn_weight: float,
    C=None,
) -> Tuple[float, float, float]:
    parsed = parse_embedding_expr(phenotype)
    if parsed is None:
        return float("inf"), float("inf"), 0.0
    x_expr, y_expr = parsed

    stress_vals = []
    knn_vals = []
    for rec in graphs:
        coords = evaluate_embedding_expr(x_expr, y_expr, rec.vars, C=C)
        if coords is None:
            return float("inf"), float("inf"), 0.0
        s = stress_loss(coords, rec.pair_i, rec.pair_j, rec.gdist)
        o = knn_overlap(coords, rec.knn_graph)
        if not np.isfinite(s) or not np.isfinite(o):
            return float("inf"), float("inf"), 0.0
        stress_vals.append(s)
        knn_vals.append(o)

    mean_stress = float(np.mean(stress_vals))
    mean_knn = float(np.mean(knn_vals))
    total = stress_weight * mean_stress + knn_weight * (1.0 - mean_knn)
    return total, mean_stress, mean_knn


def parameterize(expr: str):
    vals = []

    def _repl(m):
        vals.append(float(m.group(1)))
        return f"C[{len(vals) - 1}]"

    tpl = CONST_RE.sub(_repl, expr)
    if not vals:
        return None, None
    return tpl, np.array(vals, dtype=np.float64)


def optimize_constants(
    template: str,
    p0: np.ndarray,
    graphs: List[GraphRecord],
    stress_weight: float,
    knn_weight: float,
    n_restarts: int = 3,
) -> Tuple[np.ndarray, float]:
    def loss(params):
        val, _, _ = evaluate_on_graphs(
            template,
            graphs,
            stress_weight=stress_weight,
            knn_weight=knn_weight,
            C=params,
        )
        return val if np.isfinite(val) else 1e18

    best_params = p0.copy()
    best_loss = loss(p0)

    for i in range(n_restarts):
        try:
            if i == 0:
                x0 = p0
            else:
                x0 = p0 * (1.0 + 0.2 * np.random.randn(len(p0)))

            res = minimize(
                loss,
                x0,
                method="L-BFGS-B",
                options={"maxiter": 120, "ftol": 1e-8},
            )
            if np.isfinite(res.fun) and res.fun < best_loss:
                best_loss = float(res.fun)
                best_params = np.asarray(res.x, dtype=np.float64)
        except Exception:
            pass

    try:
        res = minimize(
            loss,
            best_params,
            method="Nelder-Mead",
            options={"maxiter": 220, "xatol": 1e-6, "fatol": 1e-8},
        )
        if np.isfinite(res.fun) and res.fun < best_loss:
            best_loss = float(res.fun)
            best_params = np.asarray(res.x, dtype=np.float64)
    except Exception:
        pass

    return best_params, best_loss


def rebuild_expr(template: str, params: np.ndarray) -> str:
    out = template
    for i, v in enumerate(params):
        out = out.replace(f"C[{i}]", f"{v:.6g}", 1)
    return out


class Objective:
    """Embedding objective with expression/template caching."""

    def __init__(
        self,
        train_graphs: List[GraphRecord],
        stress_weight: float = 1.0,
        knn_weight: float = 0.35,
        complexity_penalty=None,
        max_opt_params: int = 12,
        use_simplification: bool = True,
        max_graphs: Optional[int] = None,
        seed: int = 42,
    ):
        self.stress_weight = stress_weight
        self.knn_weight = knn_weight
        self.complexity_penalty = complexity_penalty
        self.max_opt_params = max_opt_params
        self.use_simplification = use_simplification

        self._full_graphs = train_graphs
        self._max_graphs = max_graphs
        self._seed = seed
        self._epoch = 0
        self._rng = np.random.default_rng(seed)
        self.graphs = self._sample_graphs()

        self.template_cache = {}
        self.expr_cache = {}
        self.simplified_cache = {}

    def _sample_graphs(self) -> List[GraphRecord]:
        if self._max_graphs is None or len(self._full_graphs) <= self._max_graphs:
            return list(self._full_graphs)
        idx = self._rng.choice(len(self._full_graphs), size=self._max_graphs, replace=False)
        return [self._full_graphs[int(i)] for i in idx]

    def resample(self):
        self._epoch += 1
        self._rng = np.random.default_rng(self._seed + self._epoch)
        self.graphs = self._sample_graphs()
        self.template_cache.clear()
        self.expr_cache.clear()
        self.simplified_cache.clear()

    def _penalty(self, expr: str) -> float:
        if self.complexity_penalty is None:
            return 0.0
        weight, fn = self.complexity_penalty
        return weight * fn(expr)

    def _evaluate(self, expr: str, C=None) -> float:
        val, _, _ = evaluate_on_graphs(
            expr,
            self.graphs,
            stress_weight=self.stress_weight,
            knn_weight=self.knn_weight,
            C=C,
        )
        return val

    def __call__(self, phenotype: str) -> float:
        if phenotype in self.expr_cache:
            return self.expr_cache[phenotype]

        expr = phenotype
        if self.use_simplification:
            if phenotype not in self.simplified_cache:
                self.simplified_cache[phenotype] = simplify_embedding(phenotype)
            expr = self.simplified_cache[phenotype]

        try:
            tpl, p0 = parameterize(expr)
            if tpl is None or len(p0) > self.max_opt_params:
                fit = self._evaluate(expr)
                result = fit + self._penalty(expr)
            elif tpl in self.template_cache:
                cached_p, _ = self.template_cache[tpl]
                fit = self._evaluate(tpl, C=cached_p)
                self.template_cache[tpl] = (cached_p, fit)
                result = fit + self._penalty(tpl)
            else:
                opt_p, fit = optimize_constants(
                    tpl,
                    p0,
                    self.graphs,
                    stress_weight=self.stress_weight,
                    knn_weight=self.knn_weight,
                )
                self.template_cache[tpl] = (opt_p, fit)
                result = fit + self._penalty(tpl)
        except Exception:
            result = float("inf")

        self.expr_cache[phenotype] = result
        return result
