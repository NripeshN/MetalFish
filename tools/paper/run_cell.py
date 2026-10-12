#!/usr/bin/env python3
"""Run one experiment cell from docs/paper_v2/experiments.json with fastchess.

Creates results/paper_v2/<cell_id>/ with:
  manifest.json          git SHA, binary/network/book SHA-256, tool versions, hardware, options
  fastchess_cmd.txt      the exact command line(s)
  games.pgn              all games (fastchess appends across blocks)
  engines.log            engine I/O (hybrid traces) when --engine-log is given
  therm.log              `pmset -g therm` samples every 60 s during the run
  fastchess_report.txt   fastchess's own final report(s)
  summary.json           pentanomial statistics (tools/paper/penta_stats.py) + telemetry summary

Blocks: the cell's pairs are split into N blocks (default from experiments.json
"defaults.blocks"); block b uses `-openings ... start=<offset>` so blocks draw disjoint
openings from the same seeded shuffle. Run blocks of different cells interleaved by calling
this script with --block b for each (b is 0-based); without --block all blocks run back to
back.

Examples:
    python3 tools/paper/run_cell.py RQ1a --dry-run
    python3 tools/paper/run_cell.py RQ1a --block 0 --engine-log
    python3 tools/paper/run_cell.py RQ1a --pairs 25 --tag pilot --engine-log
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Optional

REPO = pathlib.Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO / "docs" / "paper_v2" / "experiments.json"
RESULTS_ROOT = REPO / "results" / "paper_v2"


def sha256(path: pathlib.Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_capture(cmd: List[str], timeout: int = 60) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<error: {exc}>"


def uci_id(cmd: str, args: List[str]) -> str:
    try:
        p = subprocess.run(
            [cmd, *args],
            input="uci\nquit\n",
            capture_output=True,
            text=True,
            timeout=120,
        )
        for line in p.stdout.splitlines():
            if line.startswith("id name"):
                return line[len("id name ") :].strip()
    except Exception as exc:  # noqa: BLE001
        return f"<error: {exc}>"
    return "<unknown>"


def resolve_engine(engines: Dict, name: str) -> Dict:
    e = dict(engines[name])
    if "inherits" in e:
        base = resolve_engine(engines, e.pop("inherits"))
        merged = dict(base)
        merged["options"] = {**base.get("options", {}), **e.get("options", {})}
        merged.update({k: v for k, v in e.items() if k != "options"})
        return merged
    return e


def subst(value: str) -> str:
    return value.replace("${REPO}", str(REPO))


def engine_args(
    label: str, arm: Dict, engines: Dict, hash_mb: int, extra_log: bool
) -> List[str]:
    spec = resolve_engine(engines, arm["engine"])
    cmd = subst(spec["cmd"])
    if not os.path.isabs(cmd):
        cmd = str(REPO / cmd)
    parts = [f"cmd={cmd}", f"name={label}"]
    for a in spec.get("args", []):
        parts.append(f"args={subst(a)}")
    opts = {**spec.get("options", {}), **arm.get("options", {})}
    if arm.get("threads") is not None:
        opts["Threads"] = str(arm["threads"])
    if "Hash" not in opts and not spec["cmd"].endswith("lc0"):
        opts["Hash"] = str(hash_mb)
    for k, v in opts.items():
        parts.append(f"option.{k}={subst(str(v))}")
    return parts


def arm_label(arm: Dict) -> str:
    t = arm.get("threads")
    extra = ""
    for k, v in sorted(arm.get("options", {}).items()):
        if k in ("HybridABThreads",):
            continue
        extra += f"-{k}{v}"
    return f"{arm['engine']}-T{t if t is not None else 'auto'}{extra}"


def build_command(
    cell: Dict,
    defaults: Dict,
    engines: Dict,
    out: pathlib.Path,
    pairs: int,
    start: int,
    engine_log: bool,
) -> List[str]:
    fastchess = str(REPO / defaults["manager"])
    tc = cell["tc"]
    hash_mb = int(cell.get("hash_mb", defaults["hash_mb"]))
    a, b = cell["a"], cell["b"]
    la, lb = arm_label(a), arm_label(b)
    if la == lb:
        la, lb = la + "-A", lb + "-B"
    book = cell.get("book", defaults["book"])
    book_path = book if os.path.isabs(book) else str(REPO / book)
    cmd = [
        fastchess,
        "-engine",
        *engine_args(la, a, engines, hash_mb, engine_log),
        "-engine",
        *engine_args(lb, b, engines, hash_mb, engine_log),
        "-each",
        f"tc={tc}",
        "-rounds",
        str(pairs),
        "-games",
        str(defaults["games_per_round"]),
        "-concurrency",
        str(cell.get("concurrency", defaults["concurrency"])),
        "-openings",
        f"file={book_path}",
        f"format={cell.get('book_format', defaults['book_format'])}",
        f"order={defaults['book_order']}",
        f"start={start}",
        "-srand",
        str(cell.get("seed", defaults["seed"])),
        "-draw",
        *defaults["draw"].split(),
        "-resign",
        *defaults["resign"].split(),
        "-maxmoves",
        str(defaults["maxmoves"]),
        "-pgnout",
        f"file={out / 'games.pgn'}",
        "nodes=true",
        "nps=true",
        "seldepth=true",
        "timeleft=true",
        "-config",
        f"outname={out / 'fastchess_config.json'}",
        "-report",
        *defaults["report"].split(),
        "-ratinginterval",
        str(defaults["ratinginterval"]),
    ]
    if defaults.get("repeat", True):
        cmd.append("-repeat")
    if defaults.get("recover", True):
        cmd.append("-recover")
    if engine_log:
        cmd += ["-log", f"file={out / 'engines.log'}", "engine=true"]
    return cmd


def write_manifest(
    cell: Dict,
    defaults: Dict,
    engines: Dict,
    out: pathlib.Path,
    cmd: List[str],
    pairs: int,
    block: Optional[int],
    tag: str,
) -> Dict:
    git_sha = run_capture(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    dirty = (
        run_capture(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"]
        )
        != ""
    )
    arms = {}
    for key in ("a", "b"):
        spec = resolve_engine(engines, cell[key]["engine"])
        exe = subst(spec["cmd"])
        exe_path = pathlib.Path(exe if os.path.isabs(exe) else REPO / exe)
        arms[key] = {
            "engine": cell[key]["engine"],
            "label": arm_label(cell[key]),
            "cmd": str(exe_path),
            "sha256": sha256(exe_path),
            "uci_id": uci_id(str(exe_path), [subst(a) for a in spec.get("args", [])]),
            "args": [subst(a) for a in spec.get("args", [])],
            "options": {
                **spec.get("options", {}),
                **cell[key].get("options", {}),
                **(
                    {"Threads": cell[key]["threads"]}
                    if cell[key].get("threads") is not None
                    else {}
                ),
            },
        }
    book = cell.get("book", defaults["book"])
    book_path = pathlib.Path(book if os.path.isabs(book) else REPO / book)
    nets = {
        p.name: sha256(p)
        for p in (REPO / "networks").glob("*")
        if p.is_file() and p.suffix in (".pb", ".nnue")
    }
    manifest = {
        "cell": cell,
        "defaults": defaults,
        "pairs_requested": pairs,
        "block": block,
        "tag": tag,
        "git_sha": git_sha,
        "git_dirty": dirty,
        "arms": arms,
        "networks_sha256": nets,
        "book": {"path": str(book_path), "sha256": sha256(book_path)},
        "fastchess_version": run_capture(
            [str(REPO / defaults["manager"]), "--version"]
        ),
        "hardware": {
            "cpu": run_capture(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "ncpu": run_capture(["sysctl", "-n", "hw.ncpu"]),
            "perf_cores": run_capture(["sysctl", "-n", "hw.perflevel0.physicalcpu"]),
            "mem_bytes": run_capture(["sysctl", "-n", "hw.memsize"]),
            "os": platform.platform(),
            "sw_vers": run_capture(["sw_vers", "-productVersion"]),
        },
        "command": shlex.join(cmd),
        "started": dt.datetime.now().astimezone().isoformat(),
    }
    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def start_thermal_logger(out: pathlib.Path) -> subprocess.Popen:
    script = (
        "while true; do echo \"$(date +%FT%T) $(pmset -g therm 2>/dev/null | tr '\\n' ' ')\"; "
        "sleep 60; done"
    )
    return subprocess.Popen(
        ["/bin/sh", "-c", script],
        stdout=open(out / "therm.log", "a"),
        stderr=subprocess.DEVNULL,
    )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cell_id")
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument(
        "--pairs", type=int, help="override the cell's pair count (e.g. for a pilot)"
    )
    ap.add_argument("--block", type=int, help="run only this 0-based block")
    ap.add_argument("--blocks", type=int, help="override the number of blocks")
    ap.add_argument(
        "--tag", default="", help="suffix for the results directory (e.g. pilot)"
    )
    ap.add_argument(
        "--engine-log",
        action="store_true",
        help="log engine I/O (needed for hybrid traces)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    spec = json.load(open(args.manifest))
    defaults, engines = spec["defaults"], spec["engines"]
    cells = {c["id"]: c for c in spec["cells"]}
    if args.cell_id not in cells:
        print(
            f"unknown cell {args.cell_id}; known: {', '.join(cells)}", file=sys.stderr
        )
        return 2
    cell = cells[args.cell_id]
    if cell.get("type") == "gauntlet":
        print(
            "gauntlet cells are expanded by tools/paper/run_gauntlet.py (not implemented yet)",
            file=sys.stderr,
        )
        return 2
    pairs_total = args.pairs or cell["pairs"]
    nblocks = args.blocks or int(defaults.get("blocks", 1))
    per_block = -(-pairs_total // nblocks)
    out = RESULTS_ROOT / (args.cell_id + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)

    blocks = [args.block] if args.block is not None else list(range(nblocks))
    reports = []
    for b in blocks:
        start = 1 + b * per_block
        pairs = min(per_block, pairs_total - b * per_block)
        if pairs <= 0:
            continue
        cmd = build_command(cell, defaults, engines, out, pairs, start, args.engine_log)
        print(f"[{args.cell_id}] block {b}: {pairs} pairs, openings start={start}")
        print(shlex.join(cmd))
        with open(out / "fastchess_cmd.txt", "a") as fh:
            fh.write(f"# block {b}\n{shlex.join(cmd)}\n")
        if args.dry_run:
            continue
        write_manifest(cell, defaults, engines, out, cmd, pairs, b, args.tag)
        therm = start_thermal_logger(out)
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        finally:
            therm.terminate()
        elapsed = time.time() - t0
        tail = "\n".join(proc.stdout.splitlines()[-12:])
        with open(out / "fastchess_report.txt", "a") as fh:
            fh.write(
                f"# block {b} ({elapsed/3600:.2f} h, exit {proc.returncode})\n{tail}\n{proc.stderr[-2000:]}\n"
            )
        print(tail)
        reports.append(
            {"block": b, "pairs": pairs, "elapsed_s": elapsed, "exit": proc.returncode}
        )

    if args.dry_run:
        return 0
    # summary from the PGN (all blocks so far)
    sys.path.insert(0, str(REPO / "tools" / "paper"))
    import parse_hybrid_trace  # noqa: E402
    import penta_stats  # noqa: E402

    label_a = arm_label(cell["a"])
    label_b = arm_label(cell["b"])
    if label_a == label_b:
        label_a += "-A"
    games = penta_stats.read_games(str(out / "games.pgn"))
    stats = penta_stats.compute(games, label_a)
    summary = {
        "cell": args.cell_id,
        "arm_a": label_a,
        "arm_b": label_b,
        "blocks_run": reports,
        "stats": stats,
        "finished": dt.datetime.now().astimezone().isoformat(),
    }
    if (out / "engines.log").exists():
        recs = parse_hybrid_trace.parse(str(out / "engines.log"))
        summary["telemetry"] = parse_hybrid_trace.summarise(recs)
        with open(out / "moves.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(penta_stats.fmt(stats))
    if "telemetry" in summary:
        print(json.dumps(summary["telemetry"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
