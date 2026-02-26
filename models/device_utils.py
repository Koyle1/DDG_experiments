from __future__ import annotations

import torch


def _cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def _mps_available() -> bool:
    if not hasattr(torch.backends, "mps"):
        return False
    return bool(torch.backends.mps.is_available())


def resolve_device(requested: str) -> str:
    """Resolve compute device with fallback priority.

    Priority:
    - auto: cuda -> mps -> cpu
    - cuda: cuda -> mps -> cpu
    - mps: mps -> cpu
    - cpu: cpu
    """

    req = (requested or "auto").strip().lower()
    if req in ("auto", "gpu"):
        candidates = ("cuda", "mps", "cpu")
    elif req == "cuda":
        candidates = ("cuda", "mps", "cpu")
    elif req == "mps":
        candidates = ("mps", "cpu")
    elif req == "cpu":
        candidates = ("cpu",)
    else:
        raise ValueError(
            f"Unknown device '{requested}'. Expected one of: auto, cuda, mps, cpu."
        )

    availability = {
        "cuda": _cuda_available(),
        "mps": _mps_available(),
        "cpu": True,
    }
    for device_name in candidates:
        if availability[device_name]:
            return device_name
    return "cpu"
