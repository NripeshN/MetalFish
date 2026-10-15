#!/usr/bin/env python3
"""Figures and tables for the paper from the measurement artifacts (no hand-entered numbers).

Reads:
  results/paper_v2/<cell>/summary.json            experiment cells (tools/paper/run_cell.py)
  results/paper_v2/sprt/<gate>/summary.json       finished gates (tools/paper/sprt_gate.py)
  results/paper_v2/sprt/<gate>/partial_stats.json gates stopped by an author rule (penta_stats --json)
  results/paper_v2/micro/nn_latency[_frozen].json M1 (tools/paper/microbench.py nn; the frozen-build re-run wins)
  results/paper_v2/micro/ab_scaling[_frozen].json M2 (tools/paper/microbench.py ab; the frozen-build re-run wins)

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
        # A directory holding an EXCLUDED.txt (e.g. games of a superseded binary) enters nothing.
        if (
            d.is_dir()
            and s.exists()
            and not d.name.startswith("sprt")
            and not (d / "EXCLUDED.txt").exists()
        ):
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
    # cell names carry underscores (RQ1a_pilot); escape them for LaTeX text mode
    (tdir / "cells.tex").write_text("\n".join(tex).replace("_", "\\_") + "\n")
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


def pgn_game_seconds(pgn: pathlib.Path) -> float:
    """Sum of fastchess's [GameDuration "hh:mm:ss"] headers: machine time spent in games, robust to
    interrupted and resumed invocations (summary.json only records the last invocation).
    """
    total = 0.0
    if not pgn.exists():
        return total
    with open(pgn, errors="replace") as fh:
        for line in fh:
            if line.startswith("[GameDuration "):
                try:
                    h, m, sec = line.split('"')[1].split(":")
                    total += int(h) * 3600 + int(m) * 60 + float(sec)
                except (IndexError, ValueError):
                    pass
    return total


def block_ranges(manifest: dict, n_games: int, per_block: int, nblocks: int):
    """[(block, first_game, end_game)] for a cell. From the runner's invocation history when
    present (the first invocation of each block records how many games the PGN held when it
    started); otherwise sequential slices of 2 * per_block games."""
    starts = {}
    for entry in manifest.get("history") or []:
        b, g = entry.get("block"), entry.get("games_before")
        if b is not None and g is not None and b not in starts:
            starts[b] = int(g)
    if starts:
        order = sorted(starts.items(), key=lambda kv: kv[1])
        out = []
        for i, (b, first) in enumerate(order):
            end = order[i + 1][1] if i + 1 < len(order) else n_games
            out.append((b, first, end))
        return out
    return [
        (b, 2 * per_block * b, min(n_games, 2 * per_block * (b + 1)))
        for b in range(nblocks)
        if 2 * per_block * b < n_games
    ]


def write_extra_tables(cells, results: pathlib.Path, out: pathlib.Path):
    """T6 compute per cell, the telemetry table (M3-M5, RQ6a) and per-block consistency."""
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    esc = lambda t: str(t).replace("_", "\\_").replace("%", "\\%")  # noqa: E731
    # --- compute (hours from the blocks actually run)
    md = [
        "| Cell | Games | Blocks | Hours | min/game |",
        "| --- | --- | --- | --- | --- |",
    ]
    tex = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Cell & Games & Blocks & Hours & min/game \\\\",
        "\\midrule",
    ]
    for c in cells:
        d = results / c["dir"]
        hours = pgn_game_seconds(d / "games.pgn") / 3600
        if not hours:
            s = load(d / "summary.json")
            hours = sum(b.get("elapsed_s", 0) for b in s.get("blocks_run") or []) / 3600
        try:
            hist = load(d / "manifest.json").get("history") or []
            nb = len({h.get("block") for h in hist}) or c["blocks"]
        except (OSError, ValueError):
            nb = c["blocks"]
        mpg = 60 * hours / c["games"] if c["games"] else 0
        md.append(f"| {c['cell']} | {c['games']} | {nb} | {hours:.1f} | {mpg:.2f} |")
        tex.append(
            f"{esc(c['cell'])} & {c['games']} & {nb} & {hours:.1f} & {mpg:.2f} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (tdir / "compute.md").write_text("\n".join(md) + "\n")
    (tdir / "compute.tex").write_text("\n".join(tex) + "\n")
    # --- telemetry (hybrid arms only)
    md = [
        "| Cell | Hybrid moves | GPU-active | MCTS evals/move (median) | Playouts/move (median) | Disagreement rate | MCTS move chosen on disagreement | Reasons |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    tex = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Cell & Hybrid moves & GPU-active & Evals/move & Playouts/move & Disagree & MCTS chosen \\\\",
        "\\midrule",
    ]
    for c in cells:
        t = c["telemetry"]
        if not t or t.get("gpu_active_fraction") is None:
            continue
        reasons = ", ".join(
            f"{k} {v}"
            for k, v in sorted((t.get("reasons") or {}).items(), key=lambda kv: -kv[1])
        )
        dis = t.get("disagreement_rate")
        md.append(
            f"| {c['cell']} | {t.get('moves_total')} | {t['gpu_active_fraction']*100:.0f}% | {t.get('mcts_evals_median')} | "
            f"{t.get('mcts_playouts_median')} | {dis*100:.1f}% | {t.get('mcts_move_selected_on_disagreement')} / {t.get('disagreements')} | {reasons} |"
            if dis is not None
            else f"| {c['cell']} | {t.get('moves_total')} | {t['gpu_active_fraction']*100:.0f}% | {t.get('mcts_evals_median')} | {t.get('mcts_playouts_median')} | — | — | {reasons} |"
        )
        tex.append(
            f"{esc(c['cell'])} & {t.get('moves_total')} & {t['gpu_active_fraction']*100:.0f}\\% & {t.get('mcts_evals_median')} & "
            f"{t.get('mcts_playouts_median')} & {(dis*100 if dis is not None else 0):.1f}\\% & {t.get('mcts_move_selected_on_disagreement')}/{t.get('disagreements')} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (tdir / "telemetry.md").write_text("\n".join(md) + "\n")
    (tdir / "telemetry.tex").write_text("\n".join(tex) + "\n")
    # --- per-block consistency (slices games.pgn by the cell's block size)
    sys.path.insert(0, str(REPO / "tools" / "paper"))
    import penta_stats  # noqa: E402

    try:
        nblocks = int(
            load(REPO / "docs" / "paper_v2" / "experiments.json")["defaults"].get(
                "blocks", 4
            )
        )
    except (OSError, KeyError, ValueError):
        nblocks = 4
    md = [
        "| Cell | Block | Games | Elo (95% CI) | Draw ratio |",
        "| --- | --- | --- | --- | --- |",
    ]
    tex = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Cell & Block & Games & Elo & Draws \\\\",
        "\\midrule",
    ]
    for c in cells:
        d = results / c["dir"]
        pgn, man = d / "games.pgn", d / "manifest.json"
        if not pgn.exists() or not man.exists() or not c["arm_a"]:
            continue
        try:
            pairs_total = int(load(man)["cell"]["pairs"])
        except (KeyError, TypeError, ValueError):
            continue
        per_block = -(-pairs_total // nblocks)
        games = penta_stats.read_games(str(pgn))
        for b, first, end in block_ranges(load(man), len(games), per_block, nblocks):
            chunk = games[first:end]
            if not chunk:
                continue
            st = penta_stats.compute(chunk, c["arm_a"])
            md.append(
                f"| {c['cell']} | {b} | {st.get('games')} | {fmt_pm(st.get('elo'), st.get('elo_pm'))} | {(st.get('draw_ratio') or 0)*100:.0f}% |"
            )
            tex.append(
                f"{esc(c['cell'])} & {b} & {st.get('games')} & ${fmt_pm(st.get('elo'), st.get('elo_pm')).replace('±', '\\pm')}$ & {(st.get('draw_ratio') or 0)*100:.0f}\\% \\\\"
            )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (tdir / "blocks.md").write_text("\n".join(md) + "\n")
    (tdir / "blocks.tex").write_text("\n".join(tex) + "\n")


def write_therm_table(cells, results: pathlib.Path, out: pathlib.Path):
    """M6: thermal state per cell from therm.log (`pmset -g therm` sampled every minute)."""
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    esc = lambda t: str(t).replace("_", "\\_")  # noqa: E731
    md = [
        "| Cell | Samples | Warning samples | Min CPU speed limit |",
        "| --- | --- | --- | --- |",
    ]
    tex = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Cell & Samples & Warning samples & Min CPU speed limit \\\\",
        "\\midrule",
    ]
    for c in cells:
        log = results / c["dir"] / "therm.log"
        if not log.exists():
            continue
        samples = warnings = 0
        limits = []
        for line in open(log, errors="replace"):
            if "No thermal warning" in line:
                samples += 1
            elif "thermal warning level" in line.lower():
                samples += 1
                warnings += 1
            if "CPU_Speed_Limit" in line:
                try:
                    limits.append(int(line.split("=")[-1].strip()))
                except ValueError:
                    pass
        lim = f"{min(limits)}%" if limits else "—"
        md.append(f"| {c['cell']} | {samples} | {warnings} | {lim} |")
        tex.append(
            f"{esc(c['cell'])} & {samples} & {warnings} & {lim.replace('%', '\\%')} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (tdir / "therm.md").write_text("\n".join(md) + "\n")
    (tdir / "therm.tex").write_text("\n".join(tex) + "\n")


def plot_telemetry(cells, results: pathlib.Path, out: pathlib.Path):
    """F4: GPU-active fraction per hybrid cell and the distribution of MCTS playouts per
    transformer-active move (from moves.jsonl)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        c
        for c in cells
        if (c["telemetry"] or {}).get("gpu_active_fraction") is not None
    ]
    if not rows:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    a1.bar(
        [c["cell"] for c in rows],
        [100 * c["telemetry"]["gpu_active_fraction"] for c in rows],
    )
    a1.set_ylabel("hybrid-engine moves with the transformer active (%)")
    a1.set_ylim(0, 100)
    a1.tick_params(axis="x", rotation=20, labelsize=8)
    a1.grid(alpha=0.3, axis="y")
    data, labels = [], []
    for c in rows:
        mj = results / c["dir"] / "moves.jsonl"
        if not mj.exists():
            continue
        vals = []
        with open(mj) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("gpu_active") and isinstance(
                    r.get("mcts_playouts"), (int, float)
                ):
                    vals.append(r["mcts_playouts"])
        if vals:
            data.append(vals)
            labels.append(c["cell"])
    if data:
        a2.boxplot(data, showfliers=False)
        a2.set_xticks(range(1, len(labels) + 1))
        a2.set_xticklabels(labels, rotation=20, fontsize=8)
        a2.set_ylabel("MCTS playouts per transformer-active move")
        a2.grid(alpha=0.3, axis="y")
    fig.suptitle("Hybrid telemetry per cell")
    fig.tight_layout()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / f"f4_telemetry.{ext}", dpi=150)
    plt.close(fig)


def plot_reasons(cells, out: pathlib.Path):
    """F7a: decision-reason histogram per hybrid cell (from the trace summary)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [c for c in cells if (c["telemetry"] or {}).get("reasons")]
    if not rows:
        return
    keys = sorted({k for c in rows for k in c["telemetry"]["reasons"]})
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.8 / len(rows)
    for i, c in enumerate(rows):
        tot = sum(c["telemetry"]["reasons"].values()) or 1
        ax.bar(
            [k_i + i * width for k_i in range(len(keys))],
            [100 * c["telemetry"]["reasons"].get(k, 0) / tot for k in keys],
            width=width,
            label=c["cell"],
        )
    ax.set_xticks([k_i + 0.4 - width / 2 for k_i in range(len(keys))])
    ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("% of hybrid-engine moves")
    ax.set_title("Decision reasons per move (hybrid arm)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / f"f7_reasons.{ext}", dpi=150)
    plt.close(fig)


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
    # The frozen-build re-run (queue_after_must.sh writes *_frozen.json) takes precedence.
    nn = results / "micro" / "nn_latency_frozen.json"
    if not nn.exists():
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
    ab = results / "micro" / "ab_scaling_frozen.json"
    if not ab.exists():
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
            f"M2 — alpha-beta NPS vs threads, depth {d.get('depth')} ({d.get('hardware', '')})"
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
    write_extra_tables(cells, results, out)
    write_therm_table(cells, results, out)
    plot_forest(cells, out)
    plot_reasons(cells, out)
    plot_telemetry(cells, results, out)
    plot_micro(results, out)
    print(
        f"cells: {len(cells)}, gates: {len(gates)} -> {out/'tables'}, {out/'figures'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
