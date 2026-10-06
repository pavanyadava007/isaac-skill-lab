"""JSON helpers and repository paths."""

from __future__ import annotations

import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SETUP_LABEL = "NVIDIA L4, Isaac Sim 4.5 / Isaac Lab 2.1.1, simulation only, no real robot"


def _clean(o):
    import numpy as np

    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {**_clean(obj), "_setup": SETUP_LABEL, "_host_python": platform.python_version()}
    path.write_text(json.dumps(obj, indent=2) + "\n")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
