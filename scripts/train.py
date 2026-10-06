"""Train behaviour-cloning policies (MLP or BC-RNN) on the demo HDF5 files. No simulator needed.

    $ISAAC_PY scripts/train.py --grid scaling     # MLP, 25..800 demos x 3 seeds (DR data)
    $ISAAC_PY scripts/train.py --grid dr_vs_nodr  # MLP, 400 demos x 3 seeds on no-DR data
    $ISAAC_PY scripts/train.py --grid rnn         # BC-RNN, 400 demos x 3 seeds (DR data)

Each run writes checkpoints/<run>.pt and results/train/<run>.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from skilllab import models  # noqa: E402
from skilllab.dataset import demo_names, flat_obs, load_demos  # noqa: E402
from skilllab.io import write_json  # noqa: E402
from skilllab.task import OBS_DIM  # noqa: E402

DATA = {"dr": ROOT / "data" / "demos_dr.hdf5", "no_dr": ROOT / "data" / "demos_no_dr.hdf5"}
SEEDS = (0, 1, 2)
GRIDS = {
    "scaling": [("mlp", "dr", n, s) for n in (25, 50, 100, 200, 400, 800) for s in SEEDS],
    "dr_vs_nodr": [("mlp", "no_dr", 400, s) for s in SEEDS],
    "rnn": [("rnn", "dr", n, s) for n in (100, 400) for s in SEEDS],
}


def run_name(kind: str, data: str, n: int, seed: int) -> str:
    return f"{kind}_{data}_n{n}_s{seed}"


def subset(path: Path, n: int, seed: int) -> tuple[list[str], list[str]]:
    train = demo_names(path, "train")
    if n > len(train):
        raise ValueError(f"asked for {n} demos, only {len(train)} in the train split")
    perm = np.random.default_rng(1000 + seed).permutation(len(train))
    return sorted(train[i] for i in perm[:n]), demo_names(path, "valid")


def windows(demos: list[dict], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """All length-`horizon` windows (stride 1) of (obs, action) sequences, as in robomimic BC-RNN."""
    xs, ys = [], []
    for d in demos:
        o, a = flat_obs(d), d["actions"]
        for t in range(0, len(a) - horizon + 1):
            xs.append(o[t:t + horizon])
            ys.append(a[t:t + horizon])
    return np.stack(xs), np.stack(ys)


def train_one(kind: str, data: str, n: int, seed: int, steps: int, batch: int, lr: float, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    path = DATA[data]
    train_names, valid_names = subset(path, n, seed)
    tr, va = load_demos(path, train_names), load_demos(path, valid_names)
    model = models.build({"kind": kind, "obs_dim": OBS_DIM})
    if kind == "mlp":
        x = torch.tensor(np.concatenate([flat_obs(d) for d in tr]), device=device)
        y = torch.tensor(np.concatenate([d["actions"] for d in tr]), device=device)
        xv = torch.tensor(np.concatenate([flat_obs(d) for d in va]), device=device)
        yv = torch.tensor(np.concatenate([d["actions"] for d in va]), device=device)
    else:
        h = model.horizon
        x, y = (torch.tensor(a, device=device) for a in windows(tr, h))
        xv, yv = (torch.tensor(a, device=device) for a in windows(va, h))
    model = model.to(device)
    model.obs_norm.fit(x.reshape(-1, OBS_DIM))
    model.act_norm.fit(y.reshape(-1, y.shape[-1])[:, :models.CONT])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator(device=device).manual_seed(seed)

    def fwd(xb):
        return model(xb) if kind == "mlp" else model(xb)[0]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, len(x), (batch,), device=device, generator=gen)
        loss, _ = models.bc_loss(model, fwd(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        tl, tparts = models.bc_loss(model, fwd(x[: min(len(x), 20000)]), y[: min(len(y), 20000)])
        vl, vparts = models.bc_loss(model, fwd(xv), yv)
    name = run_name(kind, data, n, seed)
    meta = {"run": name, "kind": kind, "data": data, "n_demos": n, "seed": seed, "steps": steps, "batch": batch,
            "lr": lr, "train_samples": int(len(x)), "train_time_s": train_s,
            "params": int(sum(p.numel() for p in model.parameters())),
            "final_train_loss": float(tl), "valid_loss": float(vl), "valid_parts": vparts, "train_parts": tparts,
            "valid_demos": len(va), "device": torch.cuda.get_device_name(0)}
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    models.save(ROOT / "checkpoints" / f"{name}.pt", model.cpu(), meta)
    write_json(ROOT / "results" / "train" / f"{name}.json", meta)
    print(f"[train] {name}: {train_s:.1f}s, valid loss {float(vl):.4f}, grip acc {vparts['grip_acc']:.3f}", flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid", choices=sorted(GRIDS), action="append")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()
    for g in args.grid or ["scaling"]:
        for kind, data, n, seed in GRIDS[g]:
            if args.skip_existing and (ROOT / "checkpoints" / f"{run_name(kind, data, n, seed)}.pt").exists():
                continue
            train_one(kind, data, n, seed, args.steps, args.batch, args.lr, "cuda:0")


if __name__ == "__main__":
    main()
