"""Render a policy in Isaac Sim with a camera and write media/policy_rollouts.mp4 (+ a GIF preview).

Episodes come from the held-out evaluation seeds; each clip is labelled with the measured outcome.

    OMNI_KIT_ACCEPT_EULA=YES $ISAAC_PY scripts/render_video.py --headless --enable_cameras --policy mlp_dr_n800_s0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--policy", default="mlp_dr_n800_s0")
ap.add_argument("--episodes", type=int, default=4)
ap.add_argument("--seed_offset", type=int, default=0, help="first eval seed index")
ap.add_argument("--steps", type=int, default=150)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import subprocess  # noqa: E402

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

from skilllab import models  # noqa: E402
from skilllab.config import CONDITIONS, POLICY_DT  # noqa: E402
from skilllab.io import write_json  # noqa: E402
from skilllab.runner import run  # noqa: E402
from skilllab.sim.env import PickPlaceEnv  # noqa: E402
from skilllab.sim.rollout import rollout  # noqa: E402
from skilllab.task import flatten_obs  # noqa: E402

EVAL_SEED_BASE = 1_000_000


def label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    img = img.copy()
    for i, txt in enumerate(lines):
        y = 26 + 24 * i
        cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> None:
    env = PickPlaceEnv(1, CONDITIONS["nominal"].rand, size_seed=777, camera=True)
    model, meta = models.load(ROOT / "checkpoints" / f"{args.policy}.pt", env.device)
    runner = models.PolicyRunner(model, 1, env.device)
    # warm up the renderer (the first frames after start are dark)
    for _ in range(5):
        env.sim.render()
    clips, outcomes = [], []
    for k in range(args.episodes):
        seed = EVAL_SEED_BASE + args.seed_offset + k
        frames: list[np.ndarray] = []
        out = rollout(env, [seed], lambda o: runner.act(flatten_obs(o)), CONDITIONS["nominal"],
                      reset_fn=runner.reset, frame_cb=lambda t, fr=frames: fr.append(env.camera_rgb(0)),
                      steps=args.steps)
        ok, ts = bool(out["success"][0]), int(out["t_success"][0])
        ep = out["episodes"][0]
        outcomes.append({"seed": seed, "success": ok, "time_to_success_s": ts * POLICY_DT if ok else None,
                         "cube_size_m": ep["cube_size"], "cube_mass_kg": ep["cube_mass"], "friction": ep["friction"]})
        for t, fr in enumerate(frames):
            status = "placed" if ok and t >= ts else ("..." if ok or t < len(frames) - 1 else "not placed")
            clips.append(label(fr, [f"BC policy {args.policy} | eval seed {seed} | t = {t * POLICY_DT:4.1f} s",
                                    f"cube {ep['cube_size'] * 1000:.0f} mm, {ep['cube_mass'] * 1000:.0f} g, "
                                    f"mu {ep['friction']:.2f} | {status}",
                                    "Isaac Sim 4.5 / Isaac Lab 2.1.1, simulation only"]))
        print(f"[video] seed {seed}: success={ok}", flush=True)
    media = ROOT / "media"
    media.mkdir(exist_ok=True)
    mp4 = media / "policy_rollouts.mp4"
    imageio.mimwrite(mp4, clips, fps=20, codec="libx264", quality=7, pixelformat="yuv420p",
                     macro_block_size=16, ffmpeg_params=["-movflags", "+faststart"])
    gif = media / "policy_rollouts.gif"  # README preview: first two episodes, 10 fps, 360 px, palette GIF
    n_prev = len(clips) * 2 // max(1, args.episodes)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-t", f"{n_prev / 20:.2f}", "-i", str(mp4), "-vf",
                    "fps=10,scale=360:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse",
                    str(gif)], check=True)
    write_json(ROOT / "results" / "video.json", {"policy": args.policy, "episodes": outcomes, "fps": 20,
                                                   "frames": len(clips), "mp4": "media/policy_rollouts.mp4",
                                                   "gif": "media/policy_rollouts.gif",
                                                   "mp4_bytes": mp4.stat().st_size, "gif_bytes": gif.stat().st_size})


if __name__ == "__main__":
    run(main, app)
