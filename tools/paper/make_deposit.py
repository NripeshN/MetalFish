#!/usr/bin/env python3
"""Assemble the Zenodo deposit of the evaluation study (docs/paper_v2/09_handoff_goal.md, Phase A).

Collects, into <out>/ (default results/paper_v2/deposit/):
  cells/<cell>/      games.pgn, summary.json, manifest.json, moves.jsonl, therm.log,
                     fastchess_config.json, fastchess_cmd.txt, fastchess_report.txt,
                     engines.log(.gz) compressed to .gz; EXCLUDED.txt where a directory is
                     set aside (make_figures.py reads nothing from a directory that holds it)
  gates/<gate>/      summary.json or partial_stats.json, manifest.json, games.pgn (gzipped)
  micro/             *.json microbenchmark outputs
  oracle/<RQ6b|RQ6b-R>/  the oracle study: the workstation's stopped run (RQ6b) and the runners'
                     run (RQ6b-R: oracle.jsonl, its summary, stability.json, shards.json, shards/)
  micro_runner/      runner microbenchmarks (results/paper_v2/runner_micro)
  readings/          the runner cells' reading folders (results/paper_v2/runner_readings)
  runner_probe/      the runner probes the paper cites (results/paper_v2/runner_probe)
  runner/            the campaign log and state, the runner workflow and the dispatch driver
  plan/              docs/paper_v2 (protocol, audit, blueprint, evidence map, correction package,
                     manuscript source) without build products
  tools/             tools/paper/*.py and README
  bin/MAIN_READY     frozen-binary hash record; FROZEN.md if present; queue log; hostload.log
  scripts/           the queue, load-gate, oracle and logger scripts (results/paper_v2/*.sh; the
                     merge watchers are left out)
  CHECKSUMS.sha256   sha256 of every deposited file
  zenodo.json        deposit metadata (creators, description, license, related identifiers);
                     fill the arXiv identifier when it exists
Nothing is uploaded: the author creates the Zenodo record (or supplies a token to a later
upload step) after reviewing the directory. --dry-run only lists what would be copied.

The analysis scripts read the layout of results/paper_v2, not the deposit's. To regenerate the
tables from a deposit, restore that layout first: cells/<cell>/ -> <cell>/, gates/<gate>/ ->
sprt/<gate>/, oracle/<run>/ -> <run>/, micro_runner/ -> runner_micro/, readings/ ->
runner_readings/, runner/runner_campaign_* -> the top level, and decompress games.pgn.gz,
moves.jsonl.gz and engines.log.gz. The deposit's FROZEN.md has the same table.

Usage:
    python3 tools/paper/make_deposit.py [--out results/paper_v2/deposit] [--cells RQ1a,RQ2a,...]
        [--arxiv 2610.xxxxx] [--version 1.0.0] [--dry-run]

The output directory must not exist yet (or be empty), so two versions never mix.
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
    "vm_sampler.log",
    # the marker of a set-aside directory: make_figures.py reads nothing from a directory that
    # holds it, so without it a regeneration from the deposit would report the set-aside games
    "EXCLUDED.txt",
]
# the oracle study: the workstation's run (stopped at 74 positions; the stability check) and the
# runners' run (amendment 15 (a)), with the files fetch_runner_oracle.py and oracle_probe.py write
ORACLE_DIRS = ("RQ6b", "RQ6b-R")
ORACLE_FILES = (
    "oracle.jsonl",
    "oracle.jsonl.summary.json",
    "moves_input.jsonl",
    "manifest.json",
    "manifest.txt",
    "stability.json",
    "shards.json",
)
RUNNER_FILES = ("runner_campaign_log.txt", "runner_campaign_state.json")
SCRIPT_SKIP = ("automerge", "rerun_then_automerge")
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


def main(argv=None) -> int:
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
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out)
    if not args.dry_run and out.exists() and any(out.iterdir()):
        print(f"{out} exists and is not empty; choose a new --out", file=sys.stderr)
        return 2
    plan: list = []

    cells = (
        [RESULTS / c for c in args.cells.split(",")]
        if args.cells
        else [
            d
            for d in sorted(RESULTS.iterdir())
            if d.is_dir()
            and not d.name.startswith(("deposit", "runner_"))
            and d.name not in ORACLE_DIRS
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
                "venv",
            )
            and ((d / "summary.json").exists() or (d / "games.pgn").exists())
        ]
    )
    for d in cells:
        for name in CELL_FILES + ["engines.log", "engines.log.gz", "manifest.txt"]:
            src = d / name
            if src.exists():
                put(src, out / "cells" / d.name / name, plan, args.dry_run)
    for name in ORACLE_DIRS:
        d = RESULTS / name
        for f in ORACLE_FILES:
            if (d / f).exists():
                put(d / f, out / "oracle" / name / f, plan, args.dry_run)
        for src in sorted((d / "shards").rglob("*")) if (d / "shards").exists() else []:
            if src.is_file():
                put(src, out / "oracle" / name / src.relative_to(d), plan, args.dry_run)
    # the runners (amendment 15): microbenchmarks, reading folders, the probes the paper cites, the
    # campaign record, the workflow that ran every job and the driver that dispatched them
    for sub, dst in (
        ("runner_micro", "micro_runner"),
        ("runner_readings", "readings"),
        ("runner_probe", "runner_probe"),
    ):
        base = RESULTS / sub
        for src in sorted(base.rglob("*")) if base.exists() else []:
            if src.is_file() and "__pycache__" not in src.parts:
                put(src, out / dst / src.relative_to(base), plan, args.dry_run)
    for src in [RESULTS / n for n in RUNNER_FILES] + [
        REPO / ".github" / "workflows" / "paper-runner-cells.yml",
        REPO / "results" / "ars_pipeline" / "tools" / "runner_driver.py",
    ]:
        if src.exists():
            put(src, out / "runner" / src.name, plan, args.dry_run)
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
    for name in ("bin/MAIN_READY", "FROZEN.md", "queue_paper_log.txt", "hostload.log"):
        src = RESULTS / name
        if src.exists():
            put(src, out / name, plan, args.dry_run)
    # the queue, gate, oracle and logger scripts that scheduled every measurement
    for src in sorted(RESULTS.glob("*.sh")):
        if not src.name.startswith(SCRIPT_SKIP):
            put(src, out / "scripts" / src.name, plan, args.dry_run)
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
    oracle = (RESULTS / "RQ6b-R" / "oracle.jsonl").exists()
    meta = {
        "metadata": {
            "upload_type": "dataset",
            "title": "MetalFish evaluation study: protocol, frozen engine, game records, logs and analysis scripts",
            "creators": [{"name": "Niketan, Nripesh", "orcid": "0009-0008-2066-1937"}],
            "description": (
                "Complete record of the pre-registered evaluation of MetalFish's concurrent alpha-beta + transformer-MCTS "
                "search on an Apple M2 Max and on GitHub-hosted virtual Apple M1 runners: the evaluation protocol with "
                "amendments, the audit of the earlier IntelliSys 2026 chapter and the prepared Correction, every game (PGN), "
                "engine log and hybrid trace, manifests with binary, network and book hashes, thermal and runner health logs, "
                "microbenchmarks, the runner campaign record, workflow and dispatch driver, "
                + ("the oracle study, " if oracle else "")
                + "and the scripts that generate every figure and table of the paper. "
                + (
                    ""
                    if oracle
                    else "This version holds the cells of the paper version described in FROZEN.md; the oracle "
                    "study and the cells still running at its cut-off follow in a later version. "
                )
                + "Engine source: repository tag paper-v2-freeze-rc2 (the source of commit 365f36a); records and "
                f"scripts as of {git}."
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
