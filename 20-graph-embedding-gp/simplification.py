"""Symbolic simplification utilities for embedding expressions."""

import re

import sympy as sp
from sympy import Function, simplify, nsimplify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication


TRANSFORMS = standard_transformations + (implicit_multiplication,)
FEATURE_SYMBOLS = {f"f{i}": sp.symbols(f"f{i}", real=True) for i in range(5)}
NEIGH_SYMBOLS = {f"m{i}": sp.symbols(f"m{i}", real=True) for i in range(5)}

tanh_sym = Function("tanh")
sqrtabs_sym = Function("sqrtabs")
safe_div_sym = Function("safe_div")


def _split_embedding(expr: str):
    parts = [p.strip() for p in expr.split(";") if p.strip()]
    x_expr = None
    y_expr = None
    for part in parts:
        if part.startswith("x="):
            x_expr = part[2:].strip()
        elif part.startswith("y="):
            y_expr = part[2:].strip()
    return x_expr, y_expr


def _clean_expr(expr: str) -> str:
    expr = re.sub(r"\+\-", "-", expr)
    expr = re.sub(r"-\+", "-", expr)
    expr = re.sub(r"--", "+", expr)
    expr = re.sub(r"\+\+", "+", expr)
    expr = re.sub(r"\*1\.0\b", "", expr)
    expr = re.sub(r"\b1\.0\*", "", expr)
    expr = re.sub(r"\s+", "", expr)
    expr = re.sub(r"^\+", "", expr)
    return expr


def _simplify_side(expr: str) -> str:
    if not expr:
        return expr
    try:
        mapped = expr.replace("tanh", "tanh_sym").replace("safe_div", "safe_div_sym").replace("sqrtabs", "sqrtabs_sym")
        local = {
            **FEATURE_SYMBOLS,
            **NEIGH_SYMBOLS,
            "tanh_sym": tanh_sym,
            "safe_div_sym": safe_div_sym,
            "sqrtabs_sym": sqrtabs_sym,
            "log1p": sp.log,
            "abs": sp.Abs,
        }
        parsed = parse_expr(mapped, local_dict=local, transformations=TRANSFORMS)
        simp = simplify(parsed)
        simp = nsimplify(simp, rational=False, tolerance=1e-6)
        out = str(simp)
        out = out.replace("tanh_sym", "tanh").replace("safe_div_sym", "safe_div").replace("sqrtabs_sym", "sqrtabs")
        return _clean_expr(out)
    except Exception:
        return expr


def simplify_embedding(expr: str) -> str:
    """Simplify phenotype of the form x=<expr>;y=<expr>."""
    x_expr, y_expr = _split_embedding(expr)
    if x_expr is None or y_expr is None:
        return expr
    sx = _simplify_side(x_expr)
    sy = _simplify_side(y_expr)
    return f"x={sx};y={sy}"


def count_ast_nodes(expr: str) -> int:
    """Approximate complexity for parsimony pressure."""
    ops = len(re.findall(r"[+\-*/]", expr))
    funcs = len(re.findall(r"(tanh|log1p|abs|sqrtabs|safe_div)\(", expr))
    atoms = len(re.findall(r"\b(f[0-4]|m[0-4])\b|\d+\.?\d*", expr))
    return ops + funcs + atoms
