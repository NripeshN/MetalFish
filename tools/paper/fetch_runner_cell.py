#!/usr/bin/env python3
"""Join a runner cell's blocks into results/paper_v2/<cell>/ (protocol amendment 15).

Each block of a runner cell is one job of .github/workflows/paper-runner-cells.yml and leaves one
artifact, games-<cell>-b<block>, with that block's games.pgn, engines.log.zst, manifest.json,
fastchess_cmd.txt, fastchess_report.txt, therm.log, runner.json and vm_sampler.log. This tool
downloads them (`gh run download`), checks that every block of the cell is present exactly once,
complete, and played by one engine build, and writes the cell directory the workstation tools
read: games.pgn and engines.log joined in block order, manifest.json with one history entry per
block (games_before counted over the joined PGN) and each block's runner in `runner_blocks`, and
summary.json from run_cell.write_summary.

    python3 tools/paper/fetch_runner_cell.py RQ6d-R --run 12345 [--run 12399 ...]
    python3 tools/paper/fetch_runner_cell.py RQ6d-R --from-dir DIR    # artifacts already here
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import Dict, List

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "paper"))

import run_cell  # noqa: E402

BLOCK_DIR = re.compile(r"^games-(?P<cell>.+)-b(?P<block>\d+)$")
REPORT_HEAD = re.compile(r"^# block (\d+) \(([\d.]+) h, exit (-?\d+)\)$")


def count_games(pgn: pathlib.Path) -> int:
    return run_cell.count_pgn_games(pgn) if pgn.exists() else 0


def find_blocks(root: pathlib.Path, cell_id: str) -> Dict[int, List[pathlib.Path]]:
    found: Dict[int, List[pathlib.Path]] = {}
    for d in sorted(root.rglob("games-*")):
        m = BLOCK_DIR.match(d.name)
        if d.is_dir() and m and m.group("cell") == cell_id:
            found.setdefault(int(m.group("block")), []).append(d)
    return found


def join_cell(
    cell_id: str, cell: Dict, blocks: Dict[int, pathlib.Path], out: pathlib.Path
) -> Dict:
    """Write the joined cell directory from one complete directory per block; returns the
    summary (raises SystemExit on a missing, incomplete or inconsistent block)."""
    nblocks = int(cell["blocks"])
    per_block = -(-int(cell["pairs"]) // nblocks)
    missing = [b for b in range(nblocks) if b not in blocks]
    if missing:
        raise SystemExit(f"{cell_id}: blocks {missing} have no artifact")
    builds = set()
    for b in range(nblocks):
        want = 2 * min(per_block, int(cell["pairs"]) - b * per_block)
        got = count_games(blocks[b] / "games.pgn")
        if got != want:
            raise SystemExit(f"{cell_id} block {b}: {got} games, expected {want}")
        m = json.load(open(blocks[b] / "manifest.json"))
        builds.add(tuple(sorted((k, v["sha256"]) for k, v in m["arms"].items())))
    if len(builds) != 1:
        raise SystemExit(
            f"{cell_id}: blocks were played by different binaries: {builds}"
        )

    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; joined cells are written once")
    manifest: Dict = {}
    history: List[Dict] = []
    runners: List[Dict] = []
    reports: List[Dict] = []
    games_before = 0
    with open(out / "games.pgn", "w") as pgn, open(out / "engines.log", "wb") as log:
        for b in range(nblocks):
            d = blocks[b]
            pgn.write((d / "games.pgn").read_text())
            if (d / "engines.log.zst").exists():
                log.write(
                    subprocess.run(
                        ["zstd", "-dc", str(d / "engines.log.zst")],
                        capture_output=True,
                        check=True,
                    ).stdout
                )
            m = json.load(open(d / "manifest.json"))
            manifest = manifest or m
            for h in m.get("history") or []:
                history.append(
                    {**h, "games_before": games_before + (h.get("games_before") or 0)}
                )
            runner = (
                json.load(open(d / "runner.json"))
                if (d / "runner.json").exists()
                else {}
            )
            runners.append({"block": b, "artifact": d.name, **runner})
            for name in (
                "fastchess_cmd.txt",
                "fastchess_report.txt",
                "therm.log",
                "vm_sampler.log",
            ):
                if (d / name).exists():
                    with open(out / name, "a") as fh:
                        if name.endswith(".log"):
                            fh.write(f"# block {b}\n")
                        fh.write((d / name).read_text())
            for line in (d / "fastchess_report.txt").read_text().splitlines():
                r = REPORT_HEAD.match(line)
                if r:
                    reports.append(
                        {
                            "block": int(r.group(1)),
                            "elapsed_s": float(r.group(2)) * 3600,
                            "exit": int(r.group(3)),
                        }
                    )
            games_before += count_games(d / "games.pgn")
    if not (out / "engines.log").stat().st_size:
        (out / "engines.log").unlink()
    manifest.update({"history": history, "runner_blocks": runners, "block": None})
    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return run_cell.write_summary(cell_id, cell, out, reports)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cell_id")
    ap.add_argument(
        "--run", action="append", default=[], help="workflow run id (repeatable)"
    )
    ap.add_argument(
        "--from-dir",
        help="directory holding the downloaded games-<cell>-b<k> artifacts",
    )
    ap.add_argument(
        "--use",
        action="append",
        default=[],
        help="BLOCK=RUN: the run whose artifact to take for a block played more than once",
    )
    ap.add_argument("--manifest", default=str(run_cell.MANIFEST_PATH))
    args = ap.parse_args(argv)

    spec = json.load(open(args.manifest))
    cell = {c["id"]: c for c in spec["cells"]}.get(args.cell_id)
    if not cell or cell.get("machine") != "runner":
        raise SystemExit(f"{args.cell_id} is not a runner cell")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(args.from_dir or tmp)
        for run in args.run:
            subprocess.run(
                [
                    "gh",
                    "run",
                    "download",
                    run,
                    "-p",
                    f"games-{args.cell_id}-b*",
                    "-D",
                    str(root / run),
                ],
                check=True,
            )
        found = find_blocks(root, args.cell_id)
        use = dict(u.split("=", 1) for u in args.use)
        blocks = {}
        for b, dirs in found.items():
            if len(dirs) > 1:
                pick = [d for d in dirs if str(b) in use and use[str(b)] in d.parts]
                if len(pick) != 1:
                    raise SystemExit(
                        f"block {b} was played more than once ({[str(d) for d in dirs]}); "
                        f"name the run with --use {b}=RUN"
                    )
                dirs = pick
            blocks[b] = dirs[0]
        summary = join_cell(
            args.cell_id, cell, blocks, run_cell.RESULTS_ROOT / args.cell_id
        )
    print(
        f"{args.cell_id}: {summary['stats'].get('games', '?')} games joined from "
        f"{len(blocks)} blocks into {run_cell.RESULTS_ROOT / args.cell_id}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
