"""Math primitives for graph-embedding symbolic expressions."""

import re

import numpy as np


CONST_RE = re.compile(r"(?<![a-zA-Z_\d])(\d+\.?\d*)(?![a-zA-Z_\d])")
_GL = {"__builtins__": {}}


def safe_div(x, y, eps=1e-9):
    """Numerically stable elementwise division."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return x / (np.sign(y) * np.maximum(np.abs(y), eps))


def sqrtabs(x):
    """sqrt(abs(x)) helper used in grammar."""
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt(np.abs(x))


def softclip(x, lo=-20.0, hi=20.0):
    """Keep values in a finite range before expensive ops."""
    x = np.asarray(x, dtype=np.float64)
    return np.clip(x, lo, hi)
