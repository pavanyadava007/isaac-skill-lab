"""Build the static Hugging Face Space in space_build/ from results/summary.json + media, optionally upload it.

Every number on the page is read from results JSON; nothing is typed by hand.

    python scripts/build_space.py            # build only
    python scripts/build_space.py --upload   # build and upload to <user>/isaac-skill-lab (static Space)
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skilllab.io import SETUP_LABEL  # noqa: E402

OUT = ROOT / "space_build"
REPO = "pavanyadava07/isaac-skill-lab"
GITHUB = "https://github.com/pavanyadava007/isaac-skill-lab"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ci(g: dict) -> str:
    return f"{pct(g['rate'])} <span class=m>({g['successes']}/{g['n']}, CI {pct(g['ci95'][0])} to {pct(g['ci95'][1])})</span>"


def pval(p: float) -> str:
    return f"{p:.2g}" if p >= 1e-4 else f"{p:.1e}"


def build() -> None:
    s = json.loads((ROOT / "results" / "summary.json").read_text())
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    shutil.copy2(ROOT / "media" / "policy_rollouts.mp4", OUT / "policy_rollouts.mp4")
    shutil.copy2(ROOT / "media" / "scaling.svg", OUT / "scaling.svg")

    rows_scaling = "".join(f"<tr><td>{n}</td><td>{ci(g)}</td><td>{g['time_to_success_s']:.2f} s</td></tr>"
                           for n, g in s["scaling"].items())
    ex = s["expert_nominal"]
    rows_scaling += f"<tr><td>scripted expert</td><td>{ci(ex)}</td><td>{ex['time_to_success_s']:.2f} s</td></tr>"
    rows_rnn = "".join(
        f"<tr><td>{n}</td><td>{ci(d['mlp'])}</td><td>{ci(d['rnn'])}</td><td>{pval(d['paired']['p_mcnemar_exact'])}</td></tr>"
        for n, d in s["mlp_vs_rnn"].items())
    rows_dr = "".join(
        f"<tr><td>{html.escape(w)}</td><td>{ci(d['dr'])}</td><td>{ci(d['no_dr'])}</td>"
        f"<td>{pval(d['paired']['p_mcnemar_exact'])}</td></tr>" for w, d in s["dr_vs_nodr"].items())
    agents = [("scripted expert", "expert"), ("BC-MLP 800 demos", "mlp_dr_n800"), ("BC-RNN 400 demos", "rnn_dr_n400"),
              ("BC-MLP 400, no DR", "mlp_no_dr_n400")]
    conds = list(next(iter(s["robustness"].values())).keys())
    head = "".join(f"<th>{html.escape(c)}</th>" for c in conds)
    rows_rob = "".join(
        "<tr><td>" + lab + "</td>" + "".join(
            f"<td>{pct(s['robustness'][k][c]['rate'])}</td>" if s["robustness"][k].get(c) else "<td>n/a</td>"
            for c in conds) + "</tr>" for lab, k in agents)
    col = s["collection"]["dr"]
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Isaac Skill Lab</title>
<style>
:root {{ --bg:#fcfcfb; --fg:#0b0b0b; --muted:#52514e; --line:#e6e5e0; --accent:#2a78d6; --card:#ffffff; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1a1a19; --fg:#ffffff; --muted:#c3c2b7;
  --line:#3a3a37; --accent:#3987e5; --card:#232322; }} }}
:root[data-theme="dark"] {{ --bg:#1a1a19; --fg:#ffffff; --muted:#c3c2b7; --line:#3a3a37; --accent:#3987e5; --card:#232322; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:960px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:1.8rem; margin:0 0 4px; }} h2 {{ font-size:1.2rem; margin:32px 0 8px; }}
.sub, .m {{ color:var(--muted); }} .m {{ font-size:.85em; }}
.badge {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 10px; font-size:.85rem; color:var(--muted); }}
video, img {{ width:100%; height:auto; border-radius:8px; border:1px solid var(--line); background:#fcfcfb; }}
.tw {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; font-size:.92rem; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ color:var(--muted); font-weight:600; }}
a {{ color:var(--accent); }} ul {{ padding-left:20px; }}
</style></head><body><main>
<h1>Isaac Skill Lab</h1>
<p class=sub>Pick-and-place skill training and evaluation in NVIDIA Isaac Sim / Isaac Lab: own scene, Franka with a
differential-IK controller, scripted demonstrations from 1024 parallel environments, behaviour cloning, and an evaluation
harness with held-out seeds, confidence intervals and paired tests.</p>
<p><span class=badge>{html.escape(SETUP_LABEL)}</span></p>
<p>Code: <a href="{GITHUB}">{GITHUB}</a></p>
<video controls muted loop playsinline src="policy_rollouts.mp4"></video>
<p class=m>Learned BC-MLP policy on held-out seeds, rendered with an Isaac Lab camera (simulation only).</p>

<h2>Data scaling</h2>
<img src="scaling.svg" alt="success rate vs number of demonstrations">
<div class=tw><table><tr><th>demos</th><th>success (3 seeds x 200 episodes)</th><th>mean time to success</th></tr>{rows_scaling}</table></div>
<p class=m>Demos kept: {col['kept']} / {col['attempted']} expert episodes ({pct(col['kept_rate'])}), executed with DART-style noise.
Total training time for all runs: {s['training_total_min']:.1f} min.</p>

<h2>BC-MLP vs BC-RNN</h2>
<div class=tw><table><tr><th>demos</th><th>BC-MLP</th><th>BC-RNN</th><th>McNemar p</th></tr>{rows_rnn}</table></div>

<h2>Robustness (success rate)</h2>
<div class=tw><table><tr><th>agent</th>{head}</tr>{rows_rob}</table></div>
<p class=m>Shifted conditions: larger pose area, heavier and more slippery cube, 5 mm observation noise, 1 or 2 step
(50 or 100 ms) action delay, cubes smaller than in training. Policies pool 3 training seeds.</p>

<h2>Domain randomization vs none</h2>
<div class=tw><table><tr><th>test world</th><th>trained with DR</th><th>trained without DR</th><th>McNemar p</th></tr>{rows_dr}</table></div>

<h2>Limitations</h2>
<ul><li>Simulation only; no real robot, no sim-to-real claim.</li>
<li>State-based observations (exact object poses), no camera input to the policy.</li>
<li>Scripted demonstrations, not teleoperation.</li>
<li>The delta-action interface oscillates under actuation delay, for the expert and the learned policies.</li></ul>
</main></body></html>
"""
    (OUT / "index.html").write_text(page)
    (OUT / "README.md").write_text("""---
title: Isaac Skill Lab
emoji: 🤖
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: mit
short_description: Franka pick-and-place skill training in Isaac Sim
---

Static results page for https://github.com/pavanyadava007/isaac-skill-lab (simulation only).
""")
    print("built", OUT)


def upload() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(REPO, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=REPO, repo_type="space",
                      commit_message="Static results page built from results JSON")
    print("uploaded", f"https://huggingface.co/spaces/{REPO}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    build()
    if a.upload:
        upload()
