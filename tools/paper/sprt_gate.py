#!/usr/bin/env python3
"""SPRT gate for engine changes: candidate binary/options vs baseline, fastchess, UHO openings,
pentanomial statistics. Used by the strongest-engine programme (docs/paper_v2/07_engine_strength_program.md).

Examples:
    # new binary vs old binary, same options, 8 threads, 10+0.1, bounds [0, 10] nElo-model normalized
    python3 tools/paper/sprt_gate.py --name sf19port --cand build-new/metalfish --base build/metalfish \
        --tc 10+0.1 --threads 8 --hash 1024 --elo0 0 --elo1 10 --max-rounds 1500

    # same binary, candidate options (e.g. 12 threads vs 8 threads: pass --cand-threads/--base-threads)
    python3 tools/paper/sprt_gate.py --name t12 --cand build/metalfish --base build/metalfish \
        --cand-threads 12 --base-threads 8 --tc 30+0.3 --elo0 0 --elo1 8

Common options for both arms: --option NAME=VALUE; arm-specific: --cand-option / --base-option.
Results: results/paper_v2/sprt/<name>/ (games.pgn, fastchess output, summary.json).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Dict, List, Optional

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "paper"))
import penta_stats  # noqa: E402
import run_cell  # noqa: E402  (sha256, run_capture, uci_id)


def engine_block(
    name: str, cmd: str, opts: Dict[str, str], args: List[str]
) -> List[str]:
    out = [f"cmd={cmd}", f"name={name}"]
    out += [f"args={a}" for a in args]
    out += [f"option.{k}={v}" for k, v in opts.items()]
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", required=True)
    ap.add_argument("--cand", required=True, help="candidate engine binary")
    ap.add_argument("--base", required=True, help="baseline engine binary")
    ap.add_argument("--tc", default="10+0.1")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cand-threads", type=int)
    ap.add_argument("--base-threads", type=int)
    ap.add_argument("--hash", type=int, default=1024)
    ap.add_argument(
        "--move-overhead",
        type=int,
        default=50,
        help="Move Overhead (ms) for MetalFish arms",
    )
    ap.add_argument(
        "--option", action="append", default=[], help="NAME=VALUE for both arms"
    )
    ap.add_argument("--cand-option", action="append", default=[])
    ap.add_argument("--base-option", action="append", default=[])
    ap.add_argument("--cand-arg", action="append", default=[])
    ap.add_argument("--base-arg", action="append", default=[])
    ap.add_argument(
        "--book", default="reference/books/unzipped/UHO_Lichess_4852_v1.epd"
    )
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument(
        "--model", default="normalized", choices=["normalized", "logistic", "bayesian"]
    )
    ap.add_argument("--max-rounds", type=int, default=1000, help="cap on game pairs")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--engine-log", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    out = REPO / "results" / "paper_v2" / "sprt" / args.name
    out.mkdir(parents=True, exist_ok=True)
    common = dict(o.split("=", 1) for o in args.option)
    cand_opts = {**common, **dict(o.split("=", 1) for o in args.cand_option)}
    base_opts = {**common, **dict(o.split("=", 1) for o in args.base_option)}
    cand_opts.setdefault("Threads", str(args.cand_threads or args.threads))
    base_opts.setdefault("Threads", str(args.base_threads or args.threads))
    cand_opts.setdefault("Hash", str(args.hash))
    base_opts.setdefault("Hash", str(args.hash))
    for opts, exe in ((cand_opts, args.cand), (base_opts, args.base)):
        if exe.endswith("metalfish") or "metalfish" in pathlib.Path(exe).name:
            opts.setdefault("Move Overhead", str(args.move_overhead))
    cand = str(pathlib.Path(args.cand).resolve())
    base = str(pathlib.Path(args.base).resolve())
    book = args.book if os.path.isabs(args.book) else str(REPO / args.book)
    cmd = [
        str(REPO / "reference" / "fastchess" / "fastchess"),
        "-engine",
        *engine_block("cand", cand, cand_opts, args.cand_arg),
        "-engine",
        *engine_block("base", base, base_opts, args.base_arg),
        "-each",
        f"tc={args.tc}",
        "-rounds",
        str(args.max_rounds),
        "-games",
        "2",
        "-repeat",
        "-recover",
        "-concurrency",
        str(args.concurrency),
        "-openings",
        f"file={book}",
        "format=epd",
        "order=random",
        "-srand",
        str(args.seed),
        "-sprt",
        f"elo0={args.elo0}",
        f"elo1={args.elo1}",
        f"alpha={args.alpha}",
        f"beta={args.beta}",
        f"model={args.model}",
        "-draw",
        "movenumber=34",
        "movecount=8",
        "score=20",
        "-resign",
        "movecount=3",
        "score=600",
        "twosided=true",
        "-maxmoves",
        "200",
        "-pgnout",
        f"file={out / 'games.pgn'}",
        "nodes=true",
        "nps=true",
        "-config",
        f"outname={out / 'fastchess_config.json'}",
        "-report",
        "penta=true",
        "-ratinginterval",
        "20",
    ]
    if args.engine_log:
        cmd += ["-log", f"file={out / 'engines.log'}", "engine=true"]
    print(shlex.join(cmd))
    (out / "fastchess_cmd.txt").write_text(shlex.join(cmd) + "\n")
    if args.dry_run:
        return 0
    for label, exe, eargs, opts in (
        ("cand", cand, args.cand_arg, cand_opts),
        ("base", base, args.base_arg, base_opts),
    ):
        err = run_cell.preflight(exe, eargs, opts)
        if err:
            print(f"ABORT: {label} failed preflight: {err}", file=sys.stderr)
            return 3
    manifest = {
        "name": args.name,
        "started": dt.datetime.now().astimezone().isoformat(),
        "git_sha": run_cell.run_capture(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "cand": {
            "cmd": cand,
            "sha256": run_cell.sha256(pathlib.Path(cand)),
            "uci_id": run_cell.uci_id(cand, args.cand_arg),
            "options": cand_opts,
        },
        "base": {
            "cmd": base,
            "sha256": run_cell.sha256(pathlib.Path(base)),
            "uci_id": run_cell.uci_id(base, args.base_arg),
            "options": base_opts,
        },
        "tc": args.tc,
        "book": book,
        "seed": args.seed,
        "sprt": {
            "elo0": args.elo0,
            "elo1": args.elo1,
            "alpha": args.alpha,
            "beta": args.beta,
            "model": args.model,
        },
        "hardware": run_cell.run_capture(["sysctl", "-n", "machdep.cpu.brand_string"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    therm = run_cell.start_thermal_logger(out)
    lines: List[str] = []
    try:
        # Stream fastchess output to disk as it arrives so progress is visible mid-run.
        with open(out / "fastchess_output.txt", "w") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                lines.append(line.rstrip("\n"))
            proc.wait()
    finally:
        therm.terminate()
    tail = "\n".join(lines[-14:])
    print(tail)
    stats = penta_stats.compute(penta_stats.read_games(str(out / "games.pgn")), "cand")
    verdict = "unknown"
    for line in lines:
        if "SPRT" in line and ("H1" in line or "H0" in line):
            verdict = line.strip()
    summary = {
        "name": args.name,
        "fastchess_tail": tail,
        "sprt_line": verdict,
        "stats": stats,
        "finished": dt.datetime.now().astimezone().isoformat(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(penta_stats.fmt(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
