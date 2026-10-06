"""robomimic-style HDF5 demo files: data/demo_<i>/{obs/<key>, actions, rewards, dones} + mask/{train,valid}."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from skilllab.task import OBS_KEYS


def write_demos(path: str | Path, demos: list[dict], env_args: dict, valid_frac: float = 0.1, seed: int = 0) -> None:
    """demos: list of {"obs": {key: (T, d)}, "actions": (T, A), "attrs": {...}} (successful demos only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        total = 0
        names = []
        for i, d in enumerate(demos):
            name = f"demo_{i}"
            g = data.create_group(name)
            t = len(d["actions"])
            for k, v in d["obs"].items():
                g.create_dataset(f"obs/{k}", data=np.asarray(v, dtype=np.float32))
            g.create_dataset("actions", data=np.asarray(d["actions"], dtype=np.float32))
            rewards = np.zeros(t, dtype=np.float32)
            rewards[-1] = 1.0
            g.create_dataset("rewards", data=rewards)
            dones = np.zeros(t, dtype=np.int64)
            dones[-1] = 1
            g.create_dataset("dones", data=dones)
            g.attrs["num_samples"] = t
            for k, v in d.get("attrs", {}).items():
                g.attrs[k] = v
            total += t
            names.append(name)
        data.attrs["total"] = total
        data.attrs["env_args"] = json.dumps(env_args)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(names))
        n_valid = int(round(valid_frac * len(names)))
        valid = sorted(names[j] for j in order[:n_valid])
        train = sorted(names[j] for j in order[n_valid:])
        f.create_dataset("mask/train", data=np.array(train, dtype="S"))
        f.create_dataset("mask/valid", data=np.array(valid, dtype="S"))


def demo_names(path: str | Path, split: str | None = None) -> list[str]:
    with h5py.File(path, "r") as f:
        if split is None:
            names = list(f["data"].keys())
        else:
            names = [n.decode() for n in f[f"mask/{split}"][()]]
    return sorted(names, key=lambda s: int(s.split("_")[1]))


def load_demos(path: str | Path, names: list[str] | None = None) -> list[dict]:
    out = []
    with h5py.File(path, "r") as f:
        names = names if names is not None else sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        for n in names:
            g = f["data"][n]
            out.append({"name": n, "obs": {k: g["obs"][k][()] for k in g["obs"]}, "actions": g["actions"][()],
                        "attrs": dict(g.attrs)})
    return out


def env_args(path: str | Path) -> dict:
    with h5py.File(path, "r") as f:
        return json.loads(f["data"].attrs["env_args"])


def flat_obs(demo: dict) -> np.ndarray:
    return np.concatenate([demo["obs"][k] for k, _ in OBS_KEYS], axis=-1).astype(np.float32)
