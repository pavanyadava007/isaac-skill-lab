"""Generate docs/RESULTS.md, results/summary.json, media/scaling.svg and the README results block.

Every number comes from results/*.json; nothing is typed by hand. Runs with any Python 3.10+.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skilllab.io import SETUP_LABEL, read_json  # noqa: E402
from skilllab.stats import mean_std, paired_compare, wilson  # noqa: E402

R = ROOT / "results"
SEEDS = (0, 1, 2)
NS = (25, 50, 100, 200, 400, 800)
DT = 0.05


def load_eval(world: str) -> dict | None:
    p = R / "eval" / f"{world}.json"
    return read_json(p) if p.exists() else None


def group(res: dict, prefix: str, cond: str) -> dict | None:
    """Pool the 3 training seeds of one config on one condition (seed-major order, for pairing)."""
    runs = [f"{prefix}_s{s}" for s in SEEDS]
    if not all(r in res and cond in res[r] for r in runs):
        return None
    succ, times, per_seed = [], [], []
    for r in runs:
        e = res[r][cond]
        succ += e["success"]
        times += [t * DT for s, t in zip(e["success"], e["t_success"], strict=True) if s]
        per_seed.append(e["rate"])
    k, n = sum(succ), len(succ)
    lo, hi = wilson(k, n)
    tm, _ = mean_std(times)
    sm, ss = mean_std(per_seed)
    return {"successes": k, "n": n, "rate": k / n, "ci95": [lo, hi], "per_seed": per_seed,
            "seed_mean": sm, "seed_std": ss, "time_to_success_s": tm, "success_vec": succ}


def single(res: dict, run: str, cond: str) -> dict | None:
    if run not in res or cond not in res[run]:
        return None
    e = res[run][cond]
    return {"successes": e["successes"], "n": e["n"], "rate": e["rate"], "ci95": e["ci95"],
            "time_to_success_s": e["time_to_success_s"]["mean"], "success_vec": e["success"]}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def cell(g: dict | None) -> str:
    if g is None:
        return "n/a"
    return f"{pct(g['rate'])} ({g['successes']}/{g['n']}, CI {pct(g['ci95'][0])} to {pct(g['ci95'][1])})"


def tts(g: dict | None) -> str:
    if g is None or g["time_to_success_s"] != g["time_to_success_s"]:
        return "n/a"
    return f"{g['time_to_success_s']:.2f} s"


def strip(g: dict | None) -> dict | None:
    return None if g is None else {k: v for k, v in g.items() if k != "success_vec"}


def pval(p: float) -> str:
    return f"{p:.2g}" if p >= 1e-4 else f"{p:.1e}"


def scaling_svg(points: list[tuple[int, dict]], expert: dict | None) -> str:
    """Line chart of pooled success vs demos (log2 x axis) with the Wilson band; light background."""
    w, h, ml, mr, mt, mb = 640, 352, 56, 24, 52, 48
    pw, ph = w - ml - mr, h - mt - mb
    xs = [n for n, _ in points]
    import math

    lx0, lx1 = math.log2(xs[0]), math.log2(xs[-1])

    def X(n):
        return ml + (math.log2(n) - lx0) / (lx1 - lx0) * pw

    def Y(r):
        return mt + (1 - r) * ph

    blue, grey, ink, muted = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e"
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
           f'font-family="system-ui, -apple-system, Segoe UI, sans-serif" font-size="12">',
           f'<rect width="{w}" height="{h}" fill="#fcfcfb"/>',
           f'<text x="{ml}" y="22" fill="{ink}" font-size="14" font-weight="600">'
           'BC-MLP success rate vs number of demos (600 eval episodes per point)</text>']
    for r in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(r):.1f}" y2="{Y(r):.1f}" stroke="#e6e5e0"/>')
        out.append(f'<text x="{ml - 8}" y="{Y(r) + 4:.1f}" text-anchor="end" fill="{muted}">{int(r * 100)}%</text>')
    for n in xs:
        out.append(f'<text x="{X(n):.1f}" y="{h - mb + 18}" text-anchor="middle" fill="{muted}">{n}</text>')
    out.append(f'<text x="{ml + pw / 2}" y="{h - 10}" text-anchor="middle" fill="{muted}">'
               'demonstrations used for training (log scale)</text>')
    band = [(X(n), Y(g["ci95"][1])) for n, g in points] + [(X(n), Y(g["ci95"][0])) for n, g in reversed(points)]
    out.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in band) +
               f'" fill="{blue}" fill-opacity="0.15"/>')
    out.append('<polyline points="' + " ".join(f"{X(n):.1f},{Y(g['rate']):.1f}" for n, g in points) +
               f'" fill="none" stroke="{blue}" stroke-width="2"/>')
    for n, g in points:
        out.append(f'<circle cx="{X(n):.1f}" cy="{Y(g["rate"]):.1f}" r="4.5" fill="{blue}" stroke="#fcfcfb" '
                   f'stroke-width="2"><title>{n} demos: {pct(g["rate"])} '
                   f'(95% CI {pct(g["ci95"][0])} to {pct(g["ci95"][1])})</title></circle>')
        dy = 20 if g["rate"] > 0.9 else -10  # near the top, label below the point to clear the expert line
        out.append(f'<text x="{X(n):.1f}" y="{Y(g["rate"]) + dy:.1f}" text-anchor="middle" fill="{ink}">'
                   f'{100 * g["rate"]:.0f}%</text>')
    if expert is not None:
        y = Y(expert["rate"])
        out.append(f'<line x1="{ml}" x2="{w - mr}" y1="{y:.1f}" y2="{y:.1f}" stroke="{grey}" stroke-dasharray="4 4"/>')
        out.append(f'<text x="{ml + 6}" y="{y - 6:.1f}" text-anchor="start" fill="{muted}">scripted expert '
                   f'{pct(expert["rate"])}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    md: list[str] = []
    summary: dict = {"setup": SETUP_LABEL}
    add = md.append
    add("# Results\n")
    add(f"Setup: **{SETUP_LABEL}**. Generated by `scripts/make_report.py` from `results/*.json`; "
        "do not edit by hand.\n")

    # ---------------- data collection
    add("## 1. Demonstration collection (scripted expert, parallel envs)\n")
    add("| dataset | kept / attempted | kept rate | mean length (steps at 20 Hz) | expert time to success | "
        "demos with a regrasp | samples | rollout wall time |")
    add("|---|---|---|---|---|---|---|---|")
    summary["collection"] = {}
    for name in ("dr", "no_dr"):
        p = R / f"collect_{name}.json"
        if not p.exists():
            continue
        c = read_json(p)
        summary["collection"][name] = {k: c[k] for k in ("attempted", "kept", "kept_rate", "samples", "rollout_wall_s")}
        add(f"| {name} | {c['kept']} / {c['attempted']} | {pct(c['kept_rate'])} | "
            f"{c['demo_length_steps']['mean']:.1f} | {c['expert_time_to_success_s']['mean']:.2f} s | "
            f"{c['demos_with_regrasp']} | {c['samples']} | {c['rollout_wall_s']:.1f} s ({c['num_envs']} envs) |")
    add("\nExecuted actions carry Gaussian noise (DART-style) while the clean expert action is stored as the label; "
        "only successful episodes are kept, trimmed 1 s after success.\n")

    # ---------------- controller
    p = R / "controller_tuning.json"
    if p.exists():
        c = read_json(p)
        ch = c["chosen"]
        add("## 2. Controller tuning (DLS differential IK + joint PD)\n")
        add("| DLS lambda | stiffness | damping | rise 10-90% | overshoot | settled error | ramp RMS error | "
            "ramp final lag |")
        add("|---|---|---|---|---|---|---|---|")
        for r in c["rows"]:
            mark = " **(chosen)**" if (r["dls_lambda"], r["stiffness"], r["damping"]) == (
                ch["dls_lambda"], ch["arm_stiffness"], ch["arm_damping"]) else ""
            add(f"| {r['dls_lambda']}{mark} | {r['stiffness']:.0f} | {r['damping']:.0f} | "
                f"{r['rise_time_10_90_s']:.2f} s | {r['overshoot_mm']:.1f} mm | {r['settled_error_mm']:.1f} mm | "
                f"{r['ramp_rms_error_mm']:.1f} mm | {r['ramp_final_lag_mm']:.1f} mm |")
        summary["controller"] = {"chosen": ch, "rows": c["rows"], "closed_loop": c.get("closed_loop_expert", [])}
        add("")
        if c.get("closed_loop_expert"):
            add("Closed loop: scripted expert running the whole skill with each gain pair (DLS lambda "
                f"{c['closed_loop_expert'][0]['dls_lambda']}, DR ranges, no action noise):\n")
            add("| stiffness | damping | expert success (Wilson 95% CI) | mean time to success |")
            add("|---|---|---|---|")
            for r in c["closed_loop_expert"]:
                mark = " **(chosen)**" if (r["stiffness"], r["damping"]) == (ch["arm_stiffness"],
                                                                             ch["arm_damping"]) else ""
                t = r["time_to_success_s"]["mean"]
                add(f"| {r['stiffness']:.0f}{mark} | {r['damping']:.0f} | {cell(r)} | "
                    f"{'n/a' if t != t else f'{t:.2f} s'} |")
            add("\nWhy 400 / 80 although 400 / 40 tracks steps faster: see [CONTROLLER.md](CONTROLLER.md).\n")

    # ---------------- training
    tr = sorted((R / "train").glob("*.json"))
    if tr:
        add("## 3. Training\n")
        add("| config | runs | train samples (mean) | train time per run (mean) | valid loss (mean) | "
            "gripper accuracy on valid (mean) | params |")
        add("|---|---|---|---|---|---|---|")
        groups: dict[str, list[dict]] = {}
        for f in tr:
            m = read_json(f)
            groups.setdefault(re.sub(r"_s\d+$", "", m["run"]), []).append(m)
        summary["training"] = {}
        for g, ms in sorted(groups.items(), key=lambda kv: (kv[0].split("_")[0], kv[0].split("_")[1],
                                                             int(kv[0].rsplit("_n", 1)[1]))):
            t = mean_std([m["train_time_s"] for m in ms])[0]
            summary["training"][g] = {"runs": len(ms), "train_time_s": t,
                                      "valid_loss": mean_std([m["valid_loss"] for m in ms])[0]}
            add(f"| {g} | {len(ms)} | {mean_std([m['train_samples'] for m in ms])[0]:.0f} | {t:.1f} s | "
                f"{summary['training'][g]['valid_loss']:.4f} | "
                f"{pct(mean_std([m['valid_parts']['grip_acc'] for m in ms])[0])} | {ms[0]['params']} |")
        add(f"\nAll runs: {ms[0]['steps']} AdamW steps, batch {ms[0]['batch']}, cosine LR from {ms[0]['lr']}, "
            f"on {ms[0]['device']}. Total training time for all {len(tr)} runs: "
            f"{sum(read_json(f)['train_time_s'] for f in tr) / 60:.1f} min.\n")
        summary["training_total_min"] = sum(read_json(f)["train_time_s"] for f in tr) / 60

    ev = load_eval("dr")
    if ev is None:
        print("no results/eval/dr.json yet")
        return
    res = ev["results"]
    n_eval = ev["episodes"]
    exp_nom = single(res, "expert", "nominal")

    # ---------------- scaling
    add(f"## 4. Data scaling (held-out seeds, {n_eval} episodes x 3 training seeds per point)\n")
    add("| demos | success (pooled, Wilson 95% CI) | per-seed rates | mean time to success |")
    add("|---|---|---|---|")
    pts = []
    summary["scaling"] = {}
    for n in NS:
        g = group(res, f"mlp_dr_n{n}", "nominal")
        if g is None:
            continue
        pts.append((n, g))
        summary["scaling"][n] = strip(g)
        add(f"| {n} | {cell(g)} | {', '.join(pct(x) for x in g['per_seed'])} | {tts(g)} |")
    add(f"| scripted expert | {cell(exp_nom)} | - | {tts(exp_nom)} |")
    summary["expert_nominal"] = strip(exp_nom)
    add("\nPooled intervals treat the 3 x 200 episodes as independent and so do not include training-seed "
        "variance; the per-seed column shows that spread.\n")
    if pts:
        (ROOT / "media").mkdir(exist_ok=True)
        (ROOT / "media" / "scaling.svg").write_text(scaling_svg(pts, exp_nom))
        add("![scaling](../media/scaling.svg)\n")

    # ---------------- model comparison
    add("## 5. BC-MLP vs BC-RNN (same demos, same eval seeds, paired)\n")
    add("| demos | BC-MLP | BC-RNN | McNemar exact p (pooled pairs) |")
    add("|---|---|---|---|")
    summary["mlp_vs_rnn"] = {}
    for n in (100, 400):
        a, b = group(res, f"mlp_dr_n{n}", "nominal"), group(res, f"rnn_dr_n{n}", "nominal")
        if a is None or b is None:
            continue
        pc = paired_compare(a["success_vec"], b["success_vec"])
        summary["mlp_vs_rnn"][n] = {"mlp": strip(a), "rnn": strip(b), "paired": pc}
        add(f"| {n} | {cell(a)} | {cell(b)} | {pval(pc['p_mcnemar_exact'])} "
            f"(MLP-only {pc['only_a']}, RNN-only {pc['only_b']}) |")
    add("")

    # ---------------- robustness
    conds = list(ev["conditions"])
    small = load_eval("small_cubes")
    agents = [("scripted expert", "expert", False), ("BC-MLP, 400 DR demos", "mlp_dr_n400", True),
              ("BC-MLP, 800 DR demos", "mlp_dr_n800", True), ("BC-RNN, 400 DR demos", "rnn_dr_n400", True),
              ("BC-MLP, 400 no-DR demos", "mlp_no_dr_n400", True)]
    add("## 6. Robustness under shifted conditions\n")
    add("Each cell: success rate (successes/episodes, Wilson 95% CI). Policies: 3 training seeds x "
        f"{n_eval} episodes pooled; expert: {n_eval} episodes.\n")
    for c in conds:
        add(f"- `{c}`: {ev['conditions'][c]['note']}")
    if small:
        add(f"- `small_cubes`: {small['conditions']['small_cubes']['note']} (separate simulator process)")
    add("")
    cols = conds + (["small_cubes"] if small else [])
    add("| agent | " + " | ".join(cols) + " |")
    add("|---|" + "---|" * len(cols))
    summary["robustness"] = {}
    for label, key, pooled in agents:
        row = []
        summary["robustness"][key] = {}
        for c in cols:
            src = small["results"] if c == "small_cubes" else res
            g = group(src, key, c) if pooled else single(src, key, c)
            summary["robustness"][key][c] = strip(g)
            row.append(cell(g).replace(", CI", "<br>CI") if g else "n/a")
        add(f"| {label} | " + " | ".join(row) + " |")
    add("")
    add("Paired McNemar tests, nominal vs shifted condition (same seeds), BC-MLP 800 DR demos:\n")
    add("| condition | nominal-only successes | shifted-only successes | exact p |")
    add("|---|---|---|---|")
    summary["robustness_tests"] = {}
    base = group(res, "mlp_dr_n800", "nominal")
    for c in conds[1:]:
        g = group(res, "mlp_dr_n800", c)
        if base is None or g is None:
            continue
        pc = paired_compare(base["success_vec"], g["success_vec"])
        summary["robustness_tests"][c] = pc
        add(f"| {c} | {pc['only_a']} | {pc['only_b']} | {pval(pc['p_mcnemar_exact'])} |")
    add("")

    # ---------------- DR vs no DR
    add("## 7. Domain randomization vs none (BC-MLP, 400 demos each, paired by seed)\n")
    add("The no-DR dataset keeps the pose randomization but fixes cube size (50 mm), mass (100 g) and "
        "friction (0.8).\n")
    add("| test world | trained with DR | trained without DR | DR-only / no-DR-only successes | McNemar exact p |")
    add("|---|---|---|---|---|")
    nodr = load_eval("no_dr")
    worlds = [("DR ranges (nominal)", res, "nominal"), ("heavy and slippery", res, "heavy_slippery"),
              ("wide pose", res, "wide_pose")]
    if small:
        worlds.append(("small cubes", small["results"], "small_cubes"))
    if nodr:
        worlds.append(("no-DR world (fixed physics)", nodr["results"], "no_dr_world"))
    summary["dr_vs_nodr"] = {}
    for label, src, c in worlds:
        a, b = group(src, "mlp_dr_n400", c), group(src, "mlp_no_dr_n400", c)
        if a is None or b is None:
            continue
        pc = paired_compare(a["success_vec"], b["success_vec"])
        summary["dr_vs_nodr"][label] = {"dr": strip(a), "no_dr": strip(b), "paired": pc}
        add(f"| {label} | {cell(a)} | {cell(b)} | {pc['only_a']} / {pc['only_b']} | {pval(pc['p_mcnemar_exact'])} |")
    add("")

    # ---------------- failure modes
    p = R / "failure_modes.json"
    if p.exists():
        fm = read_json(p)["results"]
        add("## 8. Failure modes (separate diagnostic rerun on the same seeds)\n")
        add("| policy | condition | failures | never lifted | lifted, not on coaster | on coaster, not released or "
            "not resting | failures with a stalled TCP (last 2 s) |")
        add("|---|---|---|---|---|---|---|")
        summary["failure_modes"] = fm
        for pol, cs in fm.items():
            for c, d in cs.items():
                add(f"| {pol} | {c} | {d['failures']}/{d['episodes']} | {d['never_lifted']} | "
                    f"{d['lifted_not_on_coaster']} | {d['on_coaster_not_released_or_not_resting']} | "
                    f"{d['failures_stalled_last_2s']} |")
        add("\nMost BC failures are stalls before the grasp: the policy settles into a fixed point where it outputs "
            "almost no motion. Observation noise shakes it out of that point, which is why `obs_noise_5mm` "
            "scores higher than `nominal` for the weaker policies. GPU PhysX is not bit-reproducible, so failure "
            "counts in this rerun can differ by one or two episodes from section 6.\n")

    # ---------------- video
    p = R / "video.json"
    if p.exists():
        v = read_json(p)
        oks = sum(e["success"] for e in v["episodes"])
        add("## 9. Video\n")
        add(f"`{v['mp4']}`: policy `{v['policy']}` on {len(v['episodes'])} held-out seeds, rendered with an "
            f"Isaac Lab camera; {oks}/{len(v['episodes'])} of the filmed episodes succeed.\n")
        summary["video"] = {"policy": v["policy"], "episodes": len(v["episodes"]), "successes": oks}

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(md) + "\n")
    (R / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # README block
    readme = ROOT / "README.md"
    if readme.exists():
        blk = ["<!-- results:start (generated by scripts/make_report.py) -->", "",
               f"_{SETUP_LABEL}. {n_eval} held-out episodes per policy; policy rows pool 3 training seeds._", "",
               "| what | result |", "|---|---|"]
        for n, g in pts:
            blk.append(f"| BC-MLP, {n} demos | {cell(g)} |")
        blk.append(f"| scripted expert (teacher) | {cell(exp_nom)} |")
        for n, d in summary["mlp_vs_rnn"].items():
            blk.append(f"| BC-RNN, {n} demos | {cell(d['rnn'])}; vs MLP p = {pval(d['paired']['p_mcnemar_exact'])} |")
        for label, d in summary["dr_vs_nodr"].items():
            blk.append(f"| DR vs no-DR on {label} | {pct(d['dr']['rate'])} vs {pct(d['no_dr']['rate'])}, "
                       f"p = {pval(d['paired']['p_mcnemar_exact'])} |")
        r8 = summary["robustness"].get("mlp_dr_n800", {})
        for c in cols[1:]:
            if r8.get(c):
                blk.append(f"| BC-MLP 800 demos, `{c}` | {cell(r8[c])} |")
        ex = summary["robustness"].get("expert", {})
        for c in ("action_delay_1", "action_delay_2"):
            if ex.get(c):
                blk.append(f"| scripted expert, `{c}` | {cell(ex[c])} |")
        if "collection" in summary and "dr" in summary["collection"]:
            c = summary["collection"]["dr"]
            blk.append(f"| demos kept (DR) | {c['kept']} / {c['attempted']} ({pct(c['kept_rate'])}) |")
        if "training_total_min" in summary:
            blk.append(f"| total BC training time, all runs | {summary['training_total_min']:.1f} min |")
        blk += ["", "<!-- results:end -->"]
        txt = readme.read_text()
        txt = re.sub(r"<!-- results:start.*?<!-- results:end -->", "\n".join(blk), txt, flags=re.S)
        readme.write_text(txt)
    print("wrote docs/RESULTS.md, results/summary.json")


if __name__ == "__main__":
    main()
