#!/usr/bin/env python3
"""Assemble the Zenodo deposit of the evaluation study (docs/paper_v2/09_handoff_goal.md, Phase A).

Collects, into <out>/ (default results/paper_v2/deposit/):
  cells/<cell>/      games.pgn, summary.json, manifest.json, moves.jsonl, therm.log,
                     fastchess_config.json, fastchess_cmd.txt, fastchess_report.txt,
                     engines.log(.gz) compressed to .gz
  gates/<gate>/      summary.json or partial_stats.json, manifest.json, games.pgn (gzipped)
  micro/             *.json microbenchmark outputs
  oracle/            RQ6b outputs (oracle.jsonl, summary, manifest, moves_input)
  plan/              docs/paper_v2 (protocol, audit, blueprint, evidence map, correction package,
                     manuscript source) without build products
  tools/             tools/paper/*.py and README
  bin/MAIN_READY     frozen-binary hash record; FROZEN.md if present
  CHECKSUMS.sha256   sha256 of every deposited file
  zenodo.json        deposit metadata (creators, description, license, related identifiers);
                     fill the arXiv identifier when it exists
Nothing is uploaded: the author creates the Zenodo record (or supplies a token to a later
upload step) after reviewing the directory. --dry-run only lists what would be copied.

Usage:
    python3 tools/paper/make_deposit.py [--out results/paper_v2/deposit] [--cells RQ1a,RQ2a,...]
        [--arxiv 2610.xxxxx] [--version 1.0.0] [--dry-run]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "paper_v2"
CELL_FILES = [
    "games.pgn",
    "summary.json",
    "manifest.json",
    "moves.jsonl",
    "therm.log",
    "fastchess_config.json",
    "fastchess_cmd.txt",
    "fastchess_report.txt",
    "partial_stats.json",
]
COMPRESS = {"engines.log", "games.pgn", "moves.jsonl"}
PLAN_SKIP = {".bibcache", "__pycache__"}
PLAN_SKIP_SUFFIX = {
    ".aux",
    ".bbl",
    ".blg",
    ".log",
    ".out",
    ".fls",
    ".fdb_latexmk",
    ".synctex.gz",
}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def put(src: pathlib.Path, dst: pathlib.Path, plan: list, dry: bool) -> None:
    if src.name in COMPRESS and not src.name.endswith(".gz"):
        dst = dst.with_name(dst.name + ".gz")
    plan.append((src, dst))
    if dry:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix == ".gz" and src.suffix != ".gz":
        with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
    else:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=str(RESULTS / "deposit"))
    ap.add_argument(
        "--cells",
        help="comma-separated cell directories to include (default: every directory with a summary.json or games.pgn)",
    )
    ap.add_argument(
        "--arxiv", default="", help="arXiv identifier of the preprint, once known"
    )
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    plan: list = []

    cells = (
        [RESULTS / c for c in args.cells.split(",")]
        if args.cells
        else [
            d
            for d in sorted(RESULTS.iterdir())
            if d.is_dir()
            and d.name
            not in (
                "sprt",
                "micro",
                "bin",
                "deposit",
                "queue_logs",
                "private",
                "suites",
                "calibration",
                "x",
            )
            and ((d / "summary.json").exists() or (d / "games.pgn").exists())
        ]
    )
    for d in cells:
        for name in CELL_FILES + [
            "engines.log",
            "engines.log.gz",
            "oracle.jsonl",
            "oracle.jsonl.summary.json",
            "moves_input.jsonl",
            "manifest.txt",
        ]:
            src = d / name
            if src.exists():
                put(
                    src,
                    out / ("oracle" if d.name == "RQ6b" else "cells") / d.name / name,
                    plan,
                    args.dry_run,
                )
    sprt = RESULTS / "sprt"
    if sprt.exists():
        for d in sorted(sprt.iterdir()):
            if d.is_dir():
                for name in (
                    "summary.json",
                    "partial_stats.json",
                    "manifest.json",
                    "games.pgn",
                    "therm.log",
                    "fastchess_config.json",
                ):
                    src = d / name
                    if src.exists():
                        put(src, out / "gates" / d.name / name, plan, args.dry_run)
    for src in (
        sorted((RESULTS / "micro").glob("*.json"))
        if (RESULTS / "micro").exists()
        else []
    ):
        put(src, out / "micro" / src.name, plan, args.dry_run)
    for name in ("bin/MAIN_READY", "FROZEN.md", "queue_paper_log.txt"):
        src = RESULTS / name
        if src.exists():
            put(src, out / name, plan, args.dry_run)
    for src in sorted((REPO / "docs" / "paper_v2").rglob("*")):
        if (
            src.is_file()
            and not (PLAN_SKIP & set(src.parts))
            and src.suffix not in PLAN_SKIP_SUFFIX
        ):
            put(
                src,
                out / "plan" / src.relative_to(REPO / "docs" / "paper_v2"),
                plan,
                args.dry_run,
            )
    for src in sorted((REPO / "tools" / "paper").glob("*")):
        if src.is_file():
            put(src, out / "tools" / src.name, plan, args.dry_run)

    git = subprocess.run(
        ["git", "-C", str(REPO), "describe", "--tags", "--always"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    meta = {
        "metadata": {
            "upload_type": "dataset",
            "title": "MetalFish evaluation study: protocol, frozen engine, game records, logs and analysis scripts",
            "creators": [{"name": "Niketan, Nripesh", "orcid": "0009-0008-2066-1937"}],
            "description": (
                "Complete record of the pre-registered evaluation of MetalFish's concurrent alpha-beta + transformer-MCTS "
                "search on an Apple M2 Max: the evaluation protocol with amendments, the audit of the earlier IntelliSys 2026 "
                "chapter and the prepared Correction, every game (PGN), engine log and hybrid trace, manifests with binary, "
                "network and book hashes, thermal logs, microbenchmarks, the oracle study, and the scripts that generate "
                f"every figure and table of the paper. Engine source: repository tag {git}."
            ),
            "keywords": [
                "computer chess",
                "alpha-beta search",
                "Monte Carlo tree search",
                "transformer",
                "Apple Silicon",
                "engine evaluation",
                "pre-registration",
            ],
            "license": "cc-by-4.0",
            "version": args.version,
            "related_identifiers": [
                {
                    "relation": "cites",
                    "identifier": "10.1007/978-3-032-32726-0_39",
                    "resource_type": "publication-section",
                },
            ]
            + (
                [
                    {
                        "relation": "isSupplementTo",
                        "identifier": f"arXiv:{args.arxiv}",
                        "resource_type": "publication-preprint",
                    }
                ]
                if args.arxiv
                else []
            ),
        }
    }
    total = 0
    for src, dst in plan:
        total += src.stat().st_size
    print(f"{len(plan)} files, {total/1e9:.2f} GB before compression -> {out}")
    if args.dry_run:
        for src, dst in plan[:12]:
            print("  ", dst.relative_to(out))
        print("   ...")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    (out / "zenodo.json").write_text(json.dumps(meta, indent=2) + "\n")
    lines = []
    for f in sorted(
        p for p in out.rglob("*") if p.is_file() and p.name != "CHECKSUMS.sha256"
    ):
        lines.append(f"{sha256(f)}  {f.relative_to(out)}")
    (out / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(
        f"deposit written: {len(lines)} files, {size/1e9:.2f} GB; checksums in CHECKSUMS.sha256; metadata in zenodo.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
