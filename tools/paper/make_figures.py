#!/usr/bin/env python3
"""Figures and tables for the paper from the measurement artifacts (no hand-entered numbers).

Reads:
  results/paper_v2/<cell>/summary.json            experiment cells (tools/paper/run_cell.py)
  results/paper_v2/sprt/<gate>/summary.json       finished gates (tools/paper/sprt_gate.py)
  results/paper_v2/sprt/<gate>/partial_stats.json gates stopped by an author rule (penta_stats --json)
  results/paper_v2/micro/nn_latency[_frozen].json M1 (tools/paper/microbench.py nn; the frozen-build re-run wins)
  results/paper_v2/micro/ab_scaling[_frozen].json M2 (tools/paper/microbench.py ab; the frozen-build re-run wins)

Writes docs/paper_v2/figures/*.pdf|png and docs/paper_v2/tables/*.md|tex, and exports the schematics in
docs/paper_v2/diagrams/*.html to docs/paper_v2/figures/*.pdf with tools/paper/diagram_pdf.py.

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


def fmt_num(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def fmt_pm(x, pm, nd=1):
    if x is None or pm is None:
        return "—"
    return f"{x:+.{nd}f} ± {pm:.{nd}f}"


# Protocol names of the arms (02 §2) for tables and figures; fastchess labels stay in the PGNs.
DISPLAY_ARMS = {
    "RQ1a": "Hybrid(8) vs AB(8), 30+0.3",
    "RQ1a_pilot": "Hybrid(8) vs AB(8), 30+0.3 (pilot, pre-C18 footprint)",
    "RQ1b": "Hybrid(8) vs AB(8), 60+0.6",
    "RQ1c": "Hybrid(8) vs AB(8), 10+0.1",
    "RQ2a": "AB(6) vs AB(8), 30+0.3",
    "RQ2b": "Hybrid-split(8) vs AB(6), 30+0.3",
    "RQ2c": "Hybrid-split(8) vs AB(8), 30+0.3",
    "RQ5a": "MCTS(1) vs Lc0-mb1, 30+0.3",
    "RQ5c": "AB(8) vs SF19(8), 30+0.3",
    "RQ6d": "Hybrid(8) without hints and verification vs AB(8), 30+0.3",
}


def arms_text(c) -> str:
    return DISPLAY_ARMS.get(c["cell"], f"{c['arm_a']} vs {c['arm_b']}")


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
        "| Cell | Arms | Games | Elo (95% CI) | nElo | LOS | Pentanomial | Transformer enabled | MCTS evals/move |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    tex = [
        "\\begin{tabular}{llrrrrl}",
        "\\toprule",
        "Cell & Arms & Games & Elo & nElo & LOS & Transformer enabled \\\\",
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
            f"| {c['cell']} | {arms_text(c)} | {c['games']} | {fmt_pm(c['elo'], c['elo_pm'])} | {fmt_pm(c['nelo'], c['nelo_pm'])} | {c['los']*100:.1f}% | {c['ptnml']} | {gpu} | {ev if ev is not None else '—'} |"
        )
        tex.append(
            f"{c['cell']} & {arms_text(c)} & {c['games']} & ${fmt_pm(c['elo'], c['elo_pm']).replace('±', '\\pm')}$ & ${fmt_pm(c['nelo'], c['nelo_pm']).replace('±', '\\pm')}$ & {c['los']*100:.1f}\\% & {gpu.replace('%', '\\%')} \\\\"
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


def pgn_game_durations(pgn: pathlib.Path) -> tuple[float, int]:
    """Sum and count of fastchess's [GameDuration "hh:mm:ss"] headers: machine time spent in games,
    robust to interrupted and resumed invocations (summary.json only records the last invocation and
    lags the PGN while a cell runs).
    """
    total, n = 0.0, 0
    if not pgn.exists():
        return total, n
    with open(pgn, errors="replace") as fh:
        for line in fh:
            if line.startswith("[GameDuration "):
                try:
                    h, m, sec = line.split('"')[1].split(":")
                    total += int(h) * 3600 + int(m) * 60 + float(sec)
                    n += 1
                except (IndexError, ValueError):
                    pass
    return total, n


def pgn_game_seconds(pgn: pathlib.Path) -> float:
    return pgn_game_durations(pgn)[0]


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


LOAD_THRESHOLD_PCT = (
    300  # protocol amendment 8: external CPU above 3 cores oversubscribes the host
)


def parse_hostload(path: pathlib.Path):
    """[(aware datetime, ext_cpu %)] from hostload.log. Lines written before the logger carried a
    UTC offset (2026-10-15, before 12:33 local) are British Summer Time (+01:00)."""
    import datetime as dt
    import re

    samples = []
    if not path.exists():
        return samples
    bst = dt.timezone(dt.timedelta(hours=1))
    for line in open(path, errors="replace"):
        m = re.match(r"(\S+)\s.*ext_cpu=(\d+)%", line)
        if not m:
            continue
        stamp = m.group(1)
        try:
            if re.search(r"[+-]\d{4}$", stamp):
                t = dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S%z")
            else:
                t = dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=bst)
        except ValueError:
            continue
        samples.append((t, int(m.group(2))))
    samples.sort()
    return samples


def game_interval(g):
    """(start, end) aware datetimes from fastchess's GameStartTime/GameEndTime, or None."""
    import datetime as dt

    try:
        fmt = "%Y-%m-%dT%H:%M:%S %z"
        return dt.datetime.strptime(g["start_time"], fmt), dt.datetime.strptime(
            g["end_time"], fmt
        )
    except (KeyError, ValueError, TypeError):
        return None


def high_load_games(games, samples, threshold=LOAD_THRESHOLD_PCT):
    """Indices of games that overlap a logged minute with external CPU above `threshold`: a sample
    at time t covers the minute (t - 60 s, t], so it overlaps a game if start <= t <= end + 60 s.
    Games with no sample in that window (e.g. played before the logger existed) are not flagged.
    """
    import bisect
    import datetime as dt

    times = [t for t, _ in samples]
    flagged = set()
    for i, g in enumerate(games):
        iv = game_interval(g)
        if not iv:
            continue
        lo = bisect.bisect_left(times, iv[0])
        hi = bisect.bisect_right(times, iv[1] + dt.timedelta(seconds=60))
        if any(samples[j][1] > threshold for j in range(lo, hi)):
            flagged.add(i)
    return flagged


def write_sensitivity_table(cells, results: pathlib.Path, out: pathlib.Path):
    """Protocol amendment 8 (c): each cell's statistics after removing every game pair in which
    either game overlaps a logged minute of external CPU above the threshold."""
    sys.path.insert(0, str(REPO / "tools" / "paper"))
    import penta_stats  # noqa: E402

    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    esc = lambda t: str(t).replace("_", "\\_").replace("%", "\\%")  # noqa: E731
    samples = parse_hostload(results / "hostload.log")
    md = [
        f"| Cell | Pairs | Elo (all games) | Pairs removed (load > {LOAD_THRESHOLD_PCT}%) | Elo (low-load pairs) | Shift vs SE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    tex = [
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "Cell & Pairs & Elo (all) & Removed & Elo (low load) & Shift \\\\",
        "\\midrule",
    ]
    for c in cells:
        pgn = results / c["dir"] / "games.pgn"
        if not pgn.exists() or not c["arm_a"]:
            continue
        games = penta_stats.read_games(str(pgn))
        allst = penta_stats.compute(games, c["arm_a"])
        flagged = high_load_games(games, samples)
        bad_keys = {(games[i]["round"], games[i]["fen"]) for i in flagged}
        kept = [g for g in games if (g["round"], g["fen"]) not in bad_keys]
        low = penta_stats.compute(kept, c["arm_a"]) if kept else {}
        removed = allst.get("pairs", 0) - low.get("pairs", 0)
        frac = removed / allst["pairs"] if allst.get("pairs") else 0.0
        se = (allst.get("elo_pm") or 0) / 1.96
        if low.get("elo") is not None and allst.get("elo") is not None and se:
            shift = abs(low["elo"] - allst["elo"])
            flag = "larger than SE" if shift > se else "within SE"
        else:
            flag = "—"
        md.append(
            f"| {c['cell']} | {allst.get('pairs')} | {fmt_pm(allst.get('elo'), allst.get('elo_pm'))} | "
            f"{removed} ({frac*100:.0f}%) | {fmt_pm(low.get('elo'), low.get('elo_pm'))} | {flag} |"
        )
        tex.append(
            f"{esc(c['cell'])} & {allst.get('pairs')} & ${fmt_pm(allst.get('elo'), allst.get('elo_pm')).replace('±', '\\pm')}$ & "
            f"{removed} ({frac*100:.0f}\\%) & ${fmt_pm(low.get('elo'), low.get('elo_pm')).replace('±', '\\pm')}$ & {flag} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (tdir / "sensitivity.md").write_text("\n".join(md) + "\n")
    (tdir / "sensitivity.tex").write_text("\n".join(tex) + "\n")


FORFEIT_LIMIT = 0.005  # protocol §6 health limit on time forfeits per cell


def write_health_table(cells, results: pathlib.Path, out: pathlib.Path):
    """Health checks per cell from games.pgn: pairs, unpaired games, time forfeits by arm (protocol
    amendment 10) and how many of those with a host-load sample overlapped a high-load minute,
    adjudications and every other termination."""
    sys.path.insert(0, str(REPO / "tools" / "paper"))
    import penta_stats  # noqa: E402

    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    esc = lambda t: str(t).replace("_", "\\_")  # noqa: E731
    samples = parse_hostload(results / "hostload.log")
    md = [
        f"| Cell | Games | Pairs | Unpaired games | Time forfeits by arm | Forfeit rate | Forfeits in a minute above {LOAD_THRESHOLD_PCT}% (of those with a load sample) | Adjudicated | Other terminations |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    tex = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Cell & Games & Pairs & Unpaired & Forfeits (first / second arm) & Forfeit rate & In high load & Adjudicated \\\\",
        "\\midrule",
    ]
    for c in cells:
        pgn = results / c["dir"] / "games.pgn"
        if not pgn.exists() or not c["arm_a"]:
            continue
        games = penta_stats.read_games(str(pgn))
        st = penta_stats.compute(games, c["arm_a"])
        ff = {
            i
            for i, g in enumerate(games)
            if penta_stats.classify_termination(g) == "time_forfeit"
        }
        # threshold -1 flags every game with at least one sample in its window
        sampled = ff & high_load_games(games, samples, threshold=-1)
        loaded = ff & high_load_games(games, samples)
        under = f"{len(loaded)} of {len(sampled)}" if sampled else "—"
        by = st.get("time_forfeits_by") or {}
        fa, fb = by.get(c["arm_a"], 0), by.get(c["arm_b"], 0)
        names = arms_text(c).split(",")[0].split(" vs ")
        na, nb = names if len(names) == 2 else (c["arm_a"], c["arm_b"])
        term = dict(st.get("terminations") or {})
        forfeits = term.pop("time_forfeit", 0)
        adjudicated = term.pop("adjudication", 0)
        term.pop("normal", None)
        other = ", ".join(f"{k} {v}" for k, v in sorted(term.items())) or "none"
        rate = forfeits / st["games"] if st.get("games") else 0.0
        flag = " (above limit)" if rate > FORFEIT_LIMIT else ""
        md.append(
            f"| {c['cell']} | {st.get('games')} | {st.get('pairs')} | {st.get('unpaired_games')} | "
            f"{na} {fa}, {nb} {fb} | {rate*100:.1f}%{flag} | {under} | {adjudicated} | {other} |"
        )
        tex.append(
            f"{esc(c['cell'])} & {st.get('games')} & {st.get('pairs')} & {st.get('unpaired_games')} & "
            f"{fa} / {fb} & {rate*100:.1f}\\%{flag} & {under} & {adjudicated} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (tdir / "health.md").write_text("\n".join(md) + "\n")
    (tdir / "health.tex").write_text("\n".join(tex) + "\n")


def scheduling_regimes(results: pathlib.Path):
    """[(label, start)] from the queue log: the first start of the load gate (protocol amendment 8)
    and of the in-chunk watchdog at 450% (amendment 9) and at 300% (amendment 10)."""
    import datetime as dt
    import re

    log = results / "queue_paper_log.txt"
    if not log.exists():
        return []
    lines = log.read_text(errors="replace").splitlines()
    out = []
    for label, pat in (
        ("gate", r"gated MUST queue started"),
        ("gate + watchdog at 450%", r"\[watchdog\] started \(threshold 450%"),
        ("gate + watchdog at 300%", r"\[watchdog\] started \(threshold 300%"),
    ):
        for line in lines:
            if re.search(pat, line):
                try:
                    stamp = dt.datetime.strptime(line.split()[0], "%Y-%m-%dT%H:%M:%S%z")
                except ValueError:
                    continue
                out.append((label, stamp))
                break
    return out


def write_hostload_summary(cells, results: pathlib.Path, out: pathlib.Path):
    """External CPU (protocol amendment 8) over the whole logging period and over the minutes that
    overlap a game of any cell (same overlap rule as the sensitivity analysis), the latter also per
    scheduling regime, as a table and as LaTeX macros for the text."""
    import datetime as dt
    import statistics
    from bisect import bisect_left, bisect_right

    sys.path.insert(0, str(REPO / "tools" / "paper"))
    import penta_stats  # noqa: E402

    samples = parse_hostload(results / "hostload.log")
    if not samples:
        return
    times = [t for t, _ in samples]
    in_game = set()
    for c in cells:
        pgn = results / c["dir"] / "games.pgn"
        if not pgn.exists():
            continue
        for g in penta_stats.read_games(str(pgn)):
            iv = game_interval(g)
            if not iv:
                continue
            lo = bisect_left(times, iv[0])
            hi = bisect_right(times, iv[1] + dt.timedelta(seconds=60))
            in_game.update(range(lo, hi))

    def summary(vals):
        # 95th percentile rather than the maximum: ps's decaying %CPU can sum past the core count
        return {
            "n": len(vals),
            "median": statistics.median(vals) / 100 if vals else None,
            "p95": (
                statistics.quantiles(vals, n=20, method="inclusive")[18] / 100
                if len(vals) > 1
                else None
            ),
            "over": (
                sum(v > LOAD_THRESHOLD_PCT for v in vals) / len(vals) if vals else None
            ),
        }

    rows = {
        "log": summary([v for _, v in samples]),
        "game": summary([samples[i][1] for i in sorted(in_game)]),
    }
    first, last = samples[0][0].date().isoformat(), samples[-1][0].date().isoformat()
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    md = [
        f"| Minutes | Samples | Median (cores) | 95th percentile (cores) | Above {LOAD_THRESHOLD_PCT // 100} cores |",
        "| --- | --- | --- | --- | --- |",
    ]
    labelled = [
        (f"all logged ({first} to {last})", rows["log"]),
        ("overlapping a game", rows["game"]),
    ]
    regimes = scheduling_regimes(results)
    bounds = [("before the gate", None)] + regimes
    for i, (name, start) in enumerate(bounds):
        end = bounds[i + 1][1] if i + 1 < len(bounds) else None
        vals = [
            samples[j][1]
            for j in sorted(in_game)
            if (start is None or samples[j][0] >= start)
            and (end is None or samples[j][0] < end)
        ]
        labelled.append((f"overlapping a game: {name}", summary(vals)))
    for name, r in labelled:
        if r["n"]:
            p95 = f"{r['p95']:.1f}" if r["p95"] is not None else "—"
            md.append(
                f"| {name} | {r['n']} | {r['median']:.2f} | {p95} | {r['over']*100:.0f}% ({round(r['over'] * r['n'])}) |"
            )
    (tdir / "hostload.md").write_text("\n".join(md) + "\n")
    macros = [
        "% generated by tools/paper/make_figures.py from results/paper_v2/hostload.log",
        f"\\newcommand{{\\HostLogFirst}}{{{first}}}",
        f"\\newcommand{{\\HostLogLast}}{{{last}}}",
    ]
    for k, name in (("log", "Log"), ("game", "Game")):
        r = rows[k]
        # every macro is defined, as "--" when there is no sample, so the text always compiles
        num = lambda v, fmt: "--" if v is None else format(v, fmt)  # noqa: E731
        macros += [
            f"\\newcommand{{\\Host{name}Samples}}{{{r['n']:,}}}".replace(",", "{,}"),
            f"\\newcommand{{\\Host{name}MedianCores}}{{{num(r['median'], '.2f')}}}",
            f"\\newcommand{{\\Host{name}PninetyfiveCores}}{{{num(r['p95'], '.1f')}}}",
            f"\\newcommand{{\\Host{name}OverPct}}{{{num(None if r['over'] is None else r['over'] * 100, '.0f')}}}",
        ]
    (tdir / "hostload_macros.tex").write_text("\n".join(macros) + "\n")


def write_extra_tables(cells, results: pathlib.Path, out: pathlib.Path):
    """T6 compute per cell, the telemetry table (M3-M5, RQ6a) and per-block consistency."""
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    esc = lambda t: str(t).replace("_", "\\_").replace("%", "\\%")  # noqa: E731
    # --- compute (hours of play: the sum of the PGN's game durations)
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
        secs, timed = pgn_game_durations(d / "games.pgn")
        hours = secs / 3600
        if not hours:
            s = load(d / "summary.json")
            hours = sum(b.get("elapsed_s", 0) for b in s.get("blocks_run") or []) / 3600
            timed = c["games"]
        try:
            hist = load(d / "manifest.json").get("history") or []
            nb = len({h.get("block") for h in hist}) or c["blocks"]
        except (OSError, ValueError):
            nb = c["blocks"]
        # per game of the same PGN the hours come from (summary.json's count lags a running cell)
        mpg = 60 * hours / timed if timed else 0
        md.append(f"| {c['cell']} | {c['games']} | {nb} | {hours:.1f} | {mpg:.2f} |")
        tex.append(
            f"{esc(c['cell'])} & {c['games']} & {nb} & {hours:.1f} & {mpg:.2f} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    (tdir / "compute.md").write_text("\n".join(md) + "\n")
    (tdir / "compute.tex").write_text("\n".join(tex) + "\n")
    # --- telemetry (hybrid arms only)
    md = [
        "| Cell | Hybrid moves | Transformer enabled | MCTS evals/move (median) | Playouts/move (median) | Disagreement rate | MCTS move chosen on disagreement | Reasons |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    tex = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Cell & Hybrid moves & Transformer enabled & Evals/move & Playouts/move & Disagree & MCTS chosen \\\\",
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
    """F4: transformer-enabled fraction per hybrid cell and the distribution of MCTS playouts per
    transformer-enabled move (from moves.jsonl)."""
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
    a1.set_ylabel("hybrid moves with the transformer enabled (%)")
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
        a2.set_ylabel("MCTS playouts per transformer-enabled move")
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
    ax.set_yticklabels([f"{c['cell']}: {arms_text(c)} (n={c['games']})" for c in rows])
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
        plt.rcParams.update({"font.size": 11})
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
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
        bb = [r for r in d.get("lc0_backendbench", []) if r.get("median_eps")]
        if bb:
            a2.plot(
                [r["batch"] for r in bb],
                [r["median_eps"] for r in bb],
                marker="^",
                ls="-.",
                label="Lc0 Metal backend (evaluations/s)",
            )
        lc = [r for r in d.get("lc0", []) if r.get("nps_steady_median")]
        if lc:
            a2.plot(
                [r["minibatch"] for r in lc],
                [r["nps_steady_median"] for r in lc],
                marker="s",
                ls="--",
                label="Lc0 Metal search (nodes/s, steady state)",
            )
        for a in (a1, a2):
            a.set_xscale("log", base=2)
            a.set_xlabel("batch size")
            a.grid(alpha=0.3)
        a1.set_ylabel("median latency per batch (ms)")
        a2.set_ylabel("evaluations / s (Lc0: search nodes / s)")
        a2.legend(fontsize=9)
        fig.suptitle("M1: BT4 latency and throughput against batch size")
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
        plt.rcParams.update({"font.size": 11})
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.plot(th, [agg[str(t)]["nps_median"] / 1e6 for t in th], marker="o")
        # the M2 Max has 8 performance cores; higher thread counts also use the 4 efficiency cores
        ax.axvline(8, color="grey", ls=":", lw=1)
        ax.text(
            8.2,
            ax.get_ylim()[0],
            " efficiency cores\n from here",
            fontsize=9,
            va="bottom",
        )
        ax.set_xlabel("alpha-beta threads")
        ax.set_ylabel("nodes / s (millions)")
        ax.set_title(f"M2: alpha-beta nodes/s, depth {d.get('depth')}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(fdir / f"m2_ab_scaling.{ext}", dpi=150)
        plt.close(fig)


def export_diagrams(out: pathlib.Path) -> None:
    """Hand-drawn schematics (docs/paper_v2/diagrams/*.html, diagram-design skill) to vector PDF.

    The export needs the diagram tooling environment (Playwright + headless Chromium), which is
    kept apart from the measurement environment; without it the diagrams are skipped with a
    warning and a `build.sh --final` build fails on the missing figure."""
    import subprocess

    htmls = sorted((REPO / "docs" / "paper_v2" / "diagrams").glob("*.html"))
    if not htmls:
        return
    py = pathlib.Path.home() / ".venvs" / "diagram-design" / "bin" / "python"
    if not py.exists():
        print(
            f"warning: {py} missing; {len(htmls)} diagram(s) not exported",
            file=sys.stderr,
        )
        return
    subprocess.run(
        [
            str(py),
            str(REPO / "tools" / "paper" / "diagram_pdf.py"),
            *map(str, htmls),
            "--out",
            str(out / "figures"),
        ],
        check=True,
    )


HOSTLOAD_RE = {
    "ext_cpu": r"ext_cpu=(\d+)%",
    "gpu_util": r"gpu_util=(\d+)%",
    "gpu_mem_mb": r"gpu_mem_mb=(\d+)",
    "mem_level": r"mem_level=(\d+)%",
    "pageouts": r"pageouts=(\d+)",
}


def parse_hostload_full(path: pathlib.Path):
    """[(aware datetime, {field: value}, top process label, top process CPU %)] from hostload.log,
    with the GPU and memory fields of amendment 12 (c) where the line carries them."""
    import datetime as dt
    import re

    out = []
    if not path.exists():
        return out
    bst = dt.timezone(dt.timedelta(hours=1))
    for line in open(path, errors="replace"):
        stamp = line.split(" ", 1)[0]
        try:
            if re.search(r"[+-]\d{4}$", stamp):
                t = dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S%z")
            else:
                t = dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=bst)
        except ValueError:
            continue
        fields = {}
        for k, pat in HOSTLOAD_RE.items():
            m = re.search(pat, line)
            if m:
                fields[k] = int(m.group(1))
        if "ext_cpu" not in fields:
            continue
        m = re.search(r"top_ext=([\d.]+) (\S+)", line)
        label, pct = None, None
        if m:
            pct = float(m.group(1))
            path_tok = m.group(2)
            label = (
                "Virtualization.framework (virtual machine)"
                if "Virtualization" in path_tok
                else path_tok.rstrip("/").split("/")[-1]
            )
        out.append((t, fields, label, pct))
    out.sort(key=lambda r: r[0])
    return out


def in_game_indices(games, times):
    """Indices of host-load samples whose minute overlaps one of `games`."""
    import bisect
    import datetime as dt

    idx = set()
    for g in games:
        iv = game_interval(g)
        if not iv:
            continue
        lo = bisect.bisect_left(times, iv[0])
        hi = bisect.bisect_right(times, iv[1] + dt.timedelta(seconds=60))
        idx.update(range(lo, hi))
    return idx


def count_log_lines(path: pathlib.Path, patterns):
    """Occurrences of each substring in a (possibly gzipped) engine log, streamed."""
    import gzip

    counts = {p: 0 for p in patterns}
    if not path.exists():
        gz = path.with_name(path.name + ".gz")
        if not gz.exists():
            return None
        path = gz
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as fh:
        for line in fh:
            for p in patterns:
                if p in line:
                    counts[p] += 1
    return counts


def write_final_analyses(cells, results: pathlib.Path, out: pathlib.Path):
    """Final-reading analyses of amendment 12 and of the review (tools/paper/robustness.py):
    load regression, forfeit and opening variants with the load-sensitivity verdict, block
    heterogeneity and the chunk bootstrap, load-log coverage, derived RQ2c, the set-aside games,
    time per move by arm, adjudications, the arbiter's flags, GPU and memory covariates, the
    outside processes and the oracle summary. Run only for the final reading (--final-analyses).
    """
    import statistics

    sys.path.insert(0, str(REPO / "tools" / "paper"))
    import penta_stats  # noqa: E402
    import robustness  # noqa: E402

    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    pmtex = lambda x, pm: f"${fmt_pm(x, pm).replace('±', '\\pm')}$"  # noqa: E731
    esc = lambda t: str(t).replace("_", "\\_").replace("%", "\\%")  # noqa: E731
    samples = parse_hostload(results / "hostload.log")
    times = [t for t, _ in samples]
    try:
        nblocks = int(
            load(REPO / "docs" / "paper_v2" / "experiments.json")["defaults"].get(
                "blocks", 4
            )
        )
    except (OSError, KeyError, ValueError):
        nblocks = 4
    pairs_by_cell = {}
    rob_md = [
        "| Cell | Primary | Zero outside load (regression) | Low-load pairs only | Forfeit pairs removed | Forfeits as draws | Without first 92 openings | Load-sensitive |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rob_tex = [
        "\\begin{tabular}{lllllll}",
        "\\toprule",
        "Cell & Primary & Zero load & Low-load pairs & No forfeit pairs & Forfeits drawn & Load-sensitive \\\\",
        "\\midrule",
    ]
    blk_md = [
        "| Cell | Block estimates | Cochran's Q (df) | p | Chunks | Chunk-bootstrap 95% interval | Load-log coverage by block |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    adj_md = [
        "| Cell | Adjudicated draws | Adjudicated wins (first arm / second arm) |",
        "| --- | --- | --- |",
    ]
    for c in cells:
        d = results / c["dir"]
        pgn, man = d / "games.pgn", d / "manifest.json"
        if not pgn.exists() or not c["arm_a"]:
            continue
        games = penta_stats.read_games(str(pgn))
        prim = penta_stats.compute(games, c["arm_a"])
        pairs = robustness.pair_records(games, c["arm_a"])
        pairs_by_cell[c["cell"]] = pairs
        se = (prim.get("elo_pm") or 0) / 1.96
        reg = robustness.load_regression(games, pairs, samples, LOAD_THRESHOLD_PCT)
        flagged = high_load_games(games, samples)
        bad = {(games[i]["round"], games[i]["fen"]) for i in flagged}
        low = penta_stats.compute(
            [g for g in games if (g["round"], g["fen"]) not in bad], c["arm_a"]
        )
        ff = robustness.forfeit_variants(games, c["arm_a"])
        variants = [
            ("zero load", reg.get("elo_at_zero_load")),
            ("low-load pairs", low.get("elo")),
            ("forfeit pairs removed", ff["removed"].get("elo")),
            ("forfeits as draws", ff["as_draws"].get("elo")),
        ]
        first92 = None
        if c["cell"] == "RQ1a":
            first92 = penta_stats.compute(
                robustness.without_first_openings(games, 92), c["arm_a"]
            )
            variants.append(("without the first 92 openings", first92.get("elo")))
        moved = [
            name
            for name, v in variants
            if v is not None
            and prim.get("elo") is not None
            and se
            and abs(v - prim["elo"]) > se
        ]
        verdict = ("yes: " + ", ".join(moved)) if moved else "no"
        rob_md.append(
            f"| {c['cell']} | {fmt_pm(prim.get('elo'), prim.get('elo_pm'))} | "
            f"{fmt_pm(reg.get('elo_at_zero_load'), reg.get('elo_at_zero_load_pm'))} "
            f"({reg['pairs_exposed']} of {reg['pairs_with_samples']} logged pairs exposed) | "
            f"{fmt_pm(low.get('elo'), low.get('elo_pm'))} | "
            f"{fmt_pm(ff['removed'].get('elo'), ff['removed'].get('elo_pm'))} | "
            f"{fmt_pm(ff['as_draws'].get('elo'), ff['as_draws'].get('elo_pm'))} | "
            f"{fmt_pm(first92.get('elo'), first92.get('elo_pm')) if first92 else '—'} | {verdict} |"
        )
        rob_tex.append(
            f"{esc(c['cell'])} & {pmtex(prim.get('elo'), prim.get('elo_pm'))} & "
            f"{pmtex(reg.get('elo_at_zero_load'), reg.get('elo_at_zero_load_pm'))} & "
            f"{pmtex(low.get('elo'), low.get('elo_pm'))} & "
            f"{pmtex(ff['removed'].get('elo'), ff['removed'].get('elo_pm'))} & "
            f"{pmtex(ff['as_draws'].get('elo'), ff['as_draws'].get('elo_pm'))} & "
            f"{'yes' if moved else 'no'} \\\\"
        )
        # blocks: Cochran's Q on the block estimates, the chunk bootstrap and load-log coverage
        blocks = [0] * len(pairs)
        ests, ses, chunk_starts = [], [], []
        try:
            m = load(man)
            pairs_total = int(m["cell"]["pairs"])
            per_block = -(-pairs_total // nblocks)
            chunk_starts = [h.get("games_before", 0) for h in m.get("history") or []]
            for b, first, end in block_ranges(m, len(games), per_block, nblocks):
                st = penta_stats.compute(games[first:end], c["arm_a"])
                if st.get("elo") is not None and st.get("elo_pm"):
                    ests.append(st["elo"])
                    ses.append(st["elo_pm"] / 1.96)
                for k, p in enumerate(pairs):
                    if first <= p["i"] < end:
                        blocks[k] = b
        except (OSError, KeyError, TypeError, ValueError):
            pass
        q = robustness.cochran_q(ests, ses)
        chunks = robustness.chunk_of_pairs(pairs, chunk_starts)
        boot = robustness.chunk_bootstrap(pairs, chunks)
        cov = robustness.load_coverage(games, pairs, samples, blocks)
        q_txt = f"{q['q']:.2f} ({q['df']})" if q else "—"
        p_txt = f"{q['p']:.3f}" if q else "—"
        n_chunks = boot["chunks"] if boot else "—"
        boot_txt = f"[{boot['ci95'][0]:+.1f}, {boot['ci95'][1]:+.1f}]" if boot else "—"
        cov_txt = ", ".join(f"{b}: {w}/{n}" for b, (w, n) in cov.items())
        est_txt = ", ".join(f"{e:+.1f}" for e in ests)
        blk_md.append(
            f"| {c['cell']} | {est_txt} | {q_txt} | {p_txt} | {n_chunks} | {boot_txt} | {cov_txt} |"
        )
        adj = robustness.adjudications(games, [c["arm_a"], c["arm_b"]])
        adj_md.append(
            f"| {c['cell']} | {adj['draws']} | {adj['wins'].get(c['arm_a'], 0)} / {adj['wins'].get(c['arm_b'], 0)} |"
        )
    rob_tex += ["\\bottomrule", "\\end{tabular}"]
    (tdir / "robustness.md").write_text("\n".join(rob_md) + "\n")
    (tdir / "robustness.tex").write_text("\n".join(rob_tex) + "\n")
    (tdir / "blocks_q.md").write_text("\n".join(blk_md) + "\n")
    (tdir / "adjudications.md").write_text("\n".join(adj_md) + "\n")

    # derived RQ2c = RQ2b + RQ2a (shared openings), and the direct cell if it was played
    md = ["| Quantity | Elo | Shared openings |", "| --- | --- | --- |"]
    if "RQ2a" in pairs_by_cell and "RQ2b" in pairs_by_cell:
        der = robustness.linear_combination(
            [(pairs_by_cell["RQ2b"], 1.0), (pairs_by_cell["RQ2a"], 1.0)]
        )
        if der:
            md.append(
                f"| RQ2c derived (RQ2b + RQ2a) | {fmt_pm(der['elo'], der['elo_pm'])} | {der['shared_openings'][0]} |"
            )
        if "RQ2c" in pairs_by_cell:
            diff = robustness.linear_combination(
                [
                    (pairs_by_cell["RQ2b"], 1.0),
                    (pairs_by_cell["RQ2a"], 1.0),
                    (pairs_by_cell["RQ2c"], -1.0),
                ]
            )
            if diff:
                md.append(
                    f"| derived minus direct RQ2c | {fmt_pm(diff['elo'], diff['elo_pm'])} | {', '.join(map(str, diff['shared_openings']))} |"
                )
    (tdir / "rq2c.md").write_text("\n".join(md) + "\n")

    # the 183 games set aside by amendment 7, on their own
    rc1 = results / "RQ1a_rc1_partial" / "games.pgn"
    if rc1.exists():
        g1 = penta_stats.read_games(str(rc1))
        names = sorted({g["white"] for g in g1} | {g["black"] for g in g1})
        hyb = next((n for n in names if "ybrid" in n), names[0] if names else None)
        if hyb:
            st = penta_stats.compute(g1, hyb)
            (tdir / "setaside.md").write_text(
                "| Games | Pairs | Unpaired | Elo (first arm) |\n| --- | --- | --- | --- |\n"
                f"| {st.get('games')} | {st.get('pairs')} | {st.get('unpaired_games')} | {fmt_pm(st.get('elo'), st.get('elo_pm'))} ({hyb}) |\n"
            )

    # time per move, depth and nps by arm (PGN move comments); hybrid moves split by the traces
    tim_md = [
        "| Cell | Arm | Moves | Mean s | Median s | p90 s | p99 s | Max s | Median depth | Median nps | Mean s by own move number |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    tim_tex = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Cell & Arm (moves) & Moves & Mean s & p99 s & Depth & Mnps \\\\",
        "\\midrule",
    ]
    flag_md = []
    for c in cells:
        d = results / c["dir"]
        pgn = d / "games.pgn"
        if not pgn.exists() or not c["arm_a"]:
            continue
        cls_of = {}
        mj = d / "moves.jsonl"
        if mj.exists() and (c["telemetry"] or {}).get("moves_total"):

            def traced(fh, table=cls_of):
                # one pass: the move class of every traced position, and the records streamed on
                for line in fh:
                    r = json.loads(line)
                    if r.get("fen"):
                        kind = (
                            "time-safety"
                            if r.get("reason") == "ab_fallback_time_safety"
                            else "transformer-enabled"
                        )
                        table[r["fen"]] = (
                            kind if table.get(r["fen"], kind) == kind else "ambiguous"
                        )
                    yield r

            with open(mj) as fh:
                fc = robustness.flag_counts(traced(fh))
            logc = count_log_lines(
                d / "engines.log",
                ["subsearch stop grace expired", "engines agree, stopping early"],
            )
            flag_md.append(
                f"## {c['cell']}\n\nTraced disagreements: {fc['disagreements']}; with root hints: "
                f"{fc['with_hints']}; first hint = MCTS move: {fc['first_hint_is_mcts_move']}; "
                f"first hint = alpha-beta move: {fc['first_hint_is_ab_move']}. Engine log: "
                + (
                    ", ".join(f"'{k}' {v}" for k, v in logc.items())
                    if logc
                    else "not found"
                )
                + ".\n\n| Flag | Disagreements on which it held |\n| --- | --- |\n"
                + "\n".join(f"| {k} | {v} |" for k, v in fc["flags"].items())
                + "\n"
            )

        def classify(m, hyb=c["arm_a"], table=cls_of):
            if m["engine"] != hyb or not table:
                return "all"
            k = table.get(m.get("fen"))
            return None if k in (None, "ambiguous") else k

        st = robustness.move_stats(
            robustness.iter_engine_moves(str(pgn), with_fen=bool(cls_of)), classify
        )
        for (eng, cls), v in sorted(st.items()):
            prof = ", ".join(f"{k}: {t:.2f}" for k, t in v["time_by_index"].items())
            label = eng if cls == "all" else f"{eng} ({cls})"
            tim_tex.append(
                f"{esc(c['cell'])} & {esc(label)} & {v['moves']} & {v['time_mean']:.2f} & "
                f"{v['time_p99']:.2f} & {v['depth_p50']:.0f} & {(v['nps_p50'] or 0) / 1e6:.1f} \\\\"
            )
            tim_md.append(
                f"| {c['cell']} | {eng}{'' if cls == 'all' else ' (' + cls + ')'} | {v['moves']} | "
                f"{v['time_mean']:.3f} | {v['time_p50']:.3f} | {v['time_p90']:.3f} | {v['time_p99']:.3f} | "
                f"{v['time_max']:.3f} | {v['depth_p50']:.0f} | {(v['nps_p50'] or 0):,.0f} | {prof} |"
            )
    tim_tex += ["\\bottomrule", "\\end{tabular}"]
    (tdir / "timing.md").write_text("\n".join(tim_md) + "\n")
    (tdir / "timing.tex").write_text("\n".join(tim_tex) + "\n")
    (tdir / "flags.md").write_text("\n".join(flag_md) + "\n")

    # outside processes and the GPU and memory covariates over in-game minutes
    full = parse_hostload_full(results / "hostload.log")
    ftimes = [r[0] for r in full]
    all_games = []
    cov_md = [
        "| Cell | In-game minutes with GPU/memory fields | GPU utilisation median (p95) | GPU memory MB median | Memory available % median (min) | Page-outs during the cell's games |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in cells:
        pgn = results / c["dir"] / "games.pgn"
        if not pgn.exists():
            continue
        games = penta_stats.read_games(str(pgn))
        all_games += games
        idx = sorted(
            i for i in in_game_indices(games, ftimes) if "gpu_util" in full[i][1]
        )
        if not idx:
            continue
        gu = sorted(full[i][1]["gpu_util"] for i in idx)
        gm = [full[i][1].get("gpu_mem_mb", 0) for i in idx]
        ml = [full[i][1].get("mem_level", 0) for i in idx]
        po = sum(
            max(0, full[i][1].get("pageouts", 0) - full[i - 1][1].get("pageouts", 0))
            for i in idx
            if i > 0 and "pageouts" in full[i - 1][1]
        )
        cov_md.append(
            f"| {c['cell']} | {len(idx)} | {statistics.median(gu):.0f}% ({robustness.quantile(gu, 0.95):.0f}%) | "
            f"{statistics.median(gm):.0f} | {statistics.median(ml):.0f}% ({min(ml)}%) | {po} |"
        )
    (tdir / "covariates.md").write_text("\n".join(cov_md) + "\n")
    hot = [
        full[i]
        for i in in_game_indices(all_games, ftimes)
        if full[i][1]["ext_cpu"] > LOAD_THRESHOLD_PCT and full[i][2]
    ]
    by = {}
    for _, _, label, pct in hot:
        by.setdefault(label, []).append(pct or 0.0)
    rows = sorted(by.items(), key=lambda kv: -len(kv[1]))[:10]
    (tdir / "outside_processes.md").write_text(
        f"In-game minutes with outside CPU above {LOAD_THRESHOLD_PCT}%: {len(hot)}. Largest outside "
        "process of each such minute:\n\n| Process | Minutes as the largest | Its median CPU % |\n| --- | --- | --- |\n"
        + "\n".join(f"| {k} | {len(v)} | {statistics.median(v):.0f} |" for k, v in rows)
        + "\n"
    )

    # oracle summary (RQ6b), written by tools/paper/oracle_probe.py
    osum = results / "RQ6b" / "oracle.jsonl.summary.json"
    if osum.exists():
        o = load(osum)

        def ci(st):
            lo, hi = st.get("ci95") or (None, None)
            return (
                f"{st['estimate']:.3f} [{lo:.3f}, {hi:.3f}]"
                if lo is not None
                else f"{st['estimate']:.3f}"
            )

        md = [
            f"Positions: {o['positions']} from {o['games']} games ({o['cluster_unit']} clusters); "
            f"{o['method']['estimates']}; {o['method']['ci95']}; {o['method'].get('expected_score', '')}.",
            "",
            "| Judge | n | AB better | MCTS better | Tie | Loss AB − MCTS (ES) | Arbiter bound (ES) | Projected Elo of the bound | Realised accuracy |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for name, j in o["judges"].items():
            st = j["stats"]
            pe = j["projected_elo_of_bound"]
            pe_ci = pe.get("ci95") or [None, None]
            md.append(
                f"| {name} | {j['n']} | {ci(st['frac_ab_better'])} | {ci(st['frac_mcts_better'])} | {ci(st['frac_tie'])} | "
                f"{ci(st['loss_es_ab_minus_mcts'])} | {ci(st['arbiter_upper_bound_es'])} | "
                f"{(pe['estimate'] or 0):.1f} [{(pe_ci[0] or 0):.1f}, {(pe_ci[1] or 0):.1f}] | {ci(st['realised_accuracy'])} |"
            )
        ag = o.get("agreement")
        if ag:
            st = ag["stats"]
            md += [
                "",
                f"Judges' agreement on {ag['n']} positions: {ci(st['frac_judges_agree'])}, Cohen's kappa "
                f"{fmt_num(ag.get('cohen_kappa_sample'), 2)}; consensus AB better {ci(st['frac_consensus_ab_better'])}, MCTS better "
                f"{ci(st['frac_consensus_mcts_better'])}, undecided {ci(st['frac_undecided'])}; bound on agreed positions "
                f"{ci(st['consensus_bound_es'])} (projected Elo {fmt_num(ag['projected_elo_of_consensus_bound']['estimate'], 1)}).",
            ]
        (tdir / "oracle.md").write_text("\n".join(md) + "\n")
        tex = [
            "\\begin{tabular}{lrlllll}",
            "\\toprule",
            "Judge & $n$ & MCTS better & AB better & Tie & Bound (ES) & Projected Elo \\\\",
            "\\midrule",
        ]
        rows = [
            (n, j["n"], j["stats"], j["projected_elo_of_bound"])
            for n, j in o["judges"].items()
        ]
        if ag:
            rows.append(
                (
                    "both agree",
                    ag["n"],
                    {
                        "frac_mcts_better": ag["stats"]["frac_consensus_mcts_better"],
                        "frac_ab_better": ag["stats"]["frac_consensus_ab_better"],
                        "frac_tie": ag["stats"]["frac_consensus_tie"],
                        "arbiter_upper_bound_es": ag["stats"]["consensus_bound_es"],
                    },
                    ag["projected_elo_of_consensus_bound"],
                )
            )
        for name, n, st, pe in rows:
            pe_ci = pe.get("ci95") or [None, None]
            tex.append(
                f"{esc(name)} & {n} & {ci(st['frac_mcts_better'])} & {ci(st['frac_ab_better'])} & "
                f"{ci(st['frac_tie'])} & {ci(st['arbiter_upper_bound_es'])} & "
                f"{fmt_num(pe['estimate'], 1)} [{fmt_num(pe_ci[0], 1)}, {fmt_num(pe_ci[1], 1)}] \\\\"
            )
        tex += ["\\bottomrule", "\\end{tabular}"]
        (tdir / "oracle.tex").write_text("\n".join(tex) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", default=str(REPO / "results" / "paper_v2"))
    ap.add_argument("--out", default=str(REPO / "docs" / "paper_v2"))
    ap.add_argument(
        "--final-analyses",
        action="store_true",
        help="also write the final-reading analyses (amendment 12; slow: parses every PGN move, "
        "trace and engine log)",
    )
    args = ap.parse_args(argv)
    results = pathlib.Path(args.results)
    out = pathlib.Path(args.out)
    cells = collect_cells(results)
    gates = collect_gates(results)
    write_tables(cells, gates, out)
    write_extra_tables(cells, results, out)
    write_therm_table(cells, results, out)
    write_sensitivity_table(cells, results, out)
    write_health_table(cells, results, out)
    write_hostload_summary(cells, results, out)
    plot_forest(cells, out)
    plot_reasons(cells, out)
    plot_telemetry(cells, results, out)
    plot_micro(results, out)
    export_diagrams(out)
    if args.final_analyses:
        write_final_analyses(cells, results, out)
    print(
        f"cells: {len(cells)}, gates: {len(gates)} -> {out/'tables'}, {out/'figures'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
