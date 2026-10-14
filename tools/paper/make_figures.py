#!/usr/bin/env python3
"""Figures and tables for the paper from the measurement artifacts (no hand-entered numbers).

Reads:
  results/paper_v2/<cell>/summary.json            experiment cells (tools/paper/run_cell.py)
  results/paper_v2/sprt/<gate>/summary.json       finished gates (tools/paper/sprt_gate.py)
  results/paper_v2/sprt/<gate>/partial_stats.json gates stopped by an author rule (penta_stats --json)
  results/paper_v2/micro/nn_latency.json          M1 (tools/paper/microbench.py nn)
  results/paper_v2/micro/ab_scaling.json          M2 (tools/paper/microbench.py ab)

Writes docs/paper_v2/figures/*.pdf|png and docs/paper_v2/tables/*.md|tex.

Usage:
    python3 tools/paper/make_figures.py [--out docs/paper_v2] [--results results/paper_v2]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]


def load(path: pathlib.Path):
    with open(path) as fh:
        return json.load(fh)


def stats_of(obj: dict) -> dict:
    return obj.get("stats", obj)


def fmt_pm(x, pm, nd=1):
    if x is None or pm is None:
        return "—"
    return f"{x:+.{nd}f} ± {pm:.{nd}f}"


def collect_cells(results: pathlib.Path):
    cells = []
    for d in sorted(results.iterdir()):
        s = d / "summary.json"
        if d.is_dir() and s.exists() and not d.name.startswith("sprt"):
            obj = load(s)
            st = stats_of(obj)
            cells.append(
                {
                    # Label by directory so tagged runs (e.g. RQ1a_pilot) stay distinct.
                    "cell": d.name,
                    "dir": d.name,
                    "arm_a": obj.get("arm_a"),
                    "arm_b": obj.get("arm_b"),
                    "games": st.get("games"),
                    "pairs": st.get("pairs"),
                    "elo": st.get("elo"),
                    "elo_pm": st.get("elo_pm"),
                    "nelo": st.get("nelo"),
                    "nelo_pm": st.get("nelo_pm"),
                    "los": st.get("los"),
                    "ptnml": st.get("ptnml_0_2"),
                    "draw_ratio": st.get("draw_ratio"),
                    "telemetry": obj.get("telemetry") or {},
                    "blocks": len(obj.get("blocks_run") or []),
                }
            )
    return cells


def collect_gates(results: pathlib.Path):
    gates = []
    sprt = results / "sprt"
    if not sprt.exists():
        return gates
    for d in sorted(sprt.iterdir()):
        if not d.is_dir():
            continue
        src = d / "summary.json"
        status = "finished"
        if not src.exists():
            src = d / "partial_stats.json"
            status = "stopped (author rule)"
        if not src.exists() or src.stat().st_size == 0:
            continue
        try:
            obj = load(src)
        except json.JSONDecodeError:
            continue
        st = stats_of(obj)
        gates.append(
            {
                "gate": d.name,
                "status": status,
                "games": st.get("games"),
                "elo": st.get("elo"),
                "elo_pm": st.get("elo_pm"),
                "nelo": st.get("nelo"),
                "nelo_pm": st.get("nelo_pm"),
                "los": st.get("los"),
                "ptnml": st.get("ptnml_0_2"),
            }
        )
    return gates


def write_tables(cells, gates, out: pathlib.Path):
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    # Cells (markdown + LaTeX)
    md = [
        "| Cell | Arms | Games | Elo (95% CI) | nElo | LOS | Pentanomial | GPU-active | MCTS evals/move |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    tex = [
        "\\begin{tabular}{llrrrrl}",
        "\\toprule",
        "Cell & Arms & Games & Elo & nElo & LOS & GPU-active \\\\",
        "\\midrule",
    ]
    for c in cells:
        t = c["telemetry"]
        gpu = (
            f"{t['gpu_active_fraction']*100:.0f}%"
            if t.get("gpu_active_fraction") is not None
            else "—"
        )
        ev = t.get("mcts_evals_median")
        md.append(
            f"| {c['cell']} | {c['arm_a']} vs {c['arm_b']} | {c['games']} | {fmt_pm(c['elo'], c['elo_pm'])} | {fmt_pm(c['nelo'], c['nelo_pm'])} | {c['los']*100:.1f}% | {c['ptnml']} | {gpu} | {ev if ev is not None else '—'} |"
        )
        tex.append(
            f"{c['cell']} & {c['arm_a']} vs {c['arm_b']} & {c['games']} & ${fmt_pm(c['elo'], c['elo_pm']).replace('±', '\\pm')}$ & ${fmt_pm(c['nelo'], c['nelo_pm']).replace('±', '\\pm')}$ & {c['los']*100:.1f}\\% & {gpu.replace('%', '\\%')} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (tdir / "cells.md").write_text("\n".join(md) + "\n")
    (tdir / "cells.tex").write_text("\n".join(tex) + "\n")
    # Gates
    md = [
        "| Gate | Status | Games | Elo (95% CI) | nElo | LOS | Pentanomial |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for g in gates:
        md.append(
            f"| `{g['gate']}` | {g['status']} | {g['games']} | {fmt_pm(g['elo'], g['elo_pm'])} | {fmt_pm(g['nelo'], g['nelo_pm'])} | {g['los']*100:.1f}% | {g['ptnml']} |"
        )
    (tdir / "gates.md").write_text("\n".join(md) + "\n")


def plot_forest(cells, out: pathlib.Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [c for c in cells if c["elo"] is not None and c["elo_pm"] is not None]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 1.5))
    y = list(range(len(rows)))[::-1]
    ax.errorbar(
        [c["elo"] for c in rows],
        y,
        xerr=[c["elo_pm"] for c in rows],
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{c['cell']}: {c['arm_a']} vs {c['arm_b']} (n={c['games']})" for c in rows]
    )
    ax.set_xlabel("Elo difference (arm A − arm B), 95% CI, pentanomial")
    fig.tight_layout()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / f"cells_forest.{ext}", dpi=150)
    plt.close(fig)


def plot_micro(results: pathlib.Path, out: pathlib.Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fdir = out / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    nn = results / "micro" / "nn_latency.json"
    if nn.exists():
        d = load(nn)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
        for prec in ("fp16", "fp32"):
            rows = [r for r in d.get("metalfish", []) if r.get("precision") == prec]
            if rows:
                a1.plot(
                    [r["batch"] for r in rows],
                    [r["median_ms"] for r in rows],
                    marker="o",
                    label=f"MetalFish MPSGraph {prec.upper()}",
                )
                a2.plot(
                    [r["batch"] for r in rows],
                    [r["median_positions_per_second"] for r in rows],
                    marker="o",
                    label=f"MetalFish MPSGraph {prec.upper()}",
                )
        lc = [r for r in d.get("lc0", []) if r.get("nps_steady_median")]
        if lc:
            a2.plot(
                [r["minibatch"] for r in lc],
                [r["nps_steady_median"] for r in lc],
                marker="s",
                ls="--",
                label="lc0 Metal (benchmark, steady-state nps)",
            )
        for a in (a1, a2):
            a.set_xscale("log", base=2)
            a.set_xlabel("batch size")
            a.grid(alpha=0.3)
        a1.set_ylabel("median latency per batch (ms)")
        a2.set_ylabel("positions / s")
        a2.legend(fontsize=8)
        fig.suptitle(f"M1 — BT4 evaluation on {d.get('hardware', 'Apple Silicon')}")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(fdir / f"m1_nn_latency.{ext}", dpi=150)
        plt.close(fig)
    ab = results / "micro" / "ab_scaling.json"
    if ab.exists():
        d = load(ab)
        agg = d.get("aggregate", {})
        th = sorted(int(k) for k in agg)
        fig, ax = plt.subplots(figsize=(5, 3.8))
        ax.plot(th, [agg[str(t)]["nps_median"] / 1e6 for t in th], marker="o")
        ax.set_xlabel("alpha-beta threads")
        ax.set_ylabel("nodes / s (millions)")
        ax.set_title(
            f"M2 — alpha-beta NPS at fixed depth {d.get('depth')} ({d.get('hardware', '')})"
        )
        ax.grid(alpha=0.3)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(fdir / f"m2_ab_scaling.{ext}", dpi=150)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", default=str(REPO / "results" / "paper_v2"))
    ap.add_argument("--out", default=str(REPO / "docs" / "paper_v2"))
    args = ap.parse_args(argv)
    results = pathlib.Path(args.results)
    out = pathlib.Path(args.out)
    cells = collect_cells(results)
    gates = collect_gates(results)
    write_tables(cells, gates, out)
    plot_forest(cells, out)
    plot_micro(results, out)
    print(
        f"cells: {len(cells)}, gates: {len(gates)} -> {out/'tables'}, {out/'figures'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
