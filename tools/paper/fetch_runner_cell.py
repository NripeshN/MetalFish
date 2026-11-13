#!/usr/bin/env python3
"""Join a runner cell's blocks into results/paper_v2/<cell>/ (protocol amendment 15).

Each block of a runner cell is one job of .github/workflows/paper-runner-cells.yml and leaves one
artifact, games-<cell>-b<block>, with that block's games.pgn, engines.log.zst, manifest.json,
fastchess_cmd.txt, fastchess_report.txt, therm.log, runner.json and vm_sampler.log. This tool
downloads them (`gh run download`), checks that every block of the cell is present exactly once,
complete, and played by one engine build, and writes the cell directory the workstation tools
read: games.pgn and engines.log joined in block order, manifest.json with one history entry per
block (games_before counted over the joined PGN) and each block's runner in `runner_blocks`, and
summary.json from run_cell.write_summary. A cell whose engines reported a GPU error on stderr is
not joined (amendment 15 (i)) unless --allow-gpu-errors names an amendment that says how those
games are treated.

    python3 tools/paper/fetch_runner_cell.py RQ6d-R          # finds the blocks' artifacts in any run
    python3 tools/paper/fetch_runner_cell.py RQ6d-R --run 12345 [--run 12399 ...]
    python3 tools/paper/fetch_runner_cell.py RQ6d-R --from-dir DIR    # artifacts already here

A block tried more than once (GitHub cancels a job no runner takes; a failed attempt is resumed
by the next) contributes its one complete attempt, which holds every game of the attempts before
it; `--use BLOCK=RUN` names the attempt when that is ambiguous. Only attempts played with the
cell's fastchess time margin (`timemargin_ms`, amendment 15 (j)) are considered; the others (the
pilot blocks played without one) are listed in the manifest's `excluded_attempts` and never joined.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "paper"))

import gpu_errors  # noqa: E402
import run_cell  # noqa: E402

BLOCK_DIR = re.compile(r"^games-(?P<cell>.+)-b(?P<block>\d+)$")
REPORT_HEAD = re.compile(r"^# block (\d+) \(([\d.]+) h, exit (-?\d+)\)$")


def count_games(pgn: pathlib.Path) -> int:
    return run_cell.count_pgn_games(pgn) if pgn.exists() else 0


PAUSES = (10, 30, 90)  # seconds before each further try of a GitHub call that failed


def gh(args: List[str], target: pathlib.Path | None = None) -> str:
    """Run a gh command, trying again after a pause when it fails (GitHub's API and artifact
    service answer with transient errors, as on 2026-11-11 and 13); a download starts each try in
    an empty target directory. The last failure is raised."""
    for pause in PAUSES + (None,):
        if target is not None and target.exists():
            shutil.rmtree(target)
        p = subprocess.run(["gh", *args], capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout
        if pause is None:
            raise subprocess.CalledProcessError(
                p.returncode, ["gh", *args], p.stdout, p.stderr
            )
        print(
            f"gh {' '.join(args[:3])} failed ({p.stderr.strip()[:200]}); again in {pause} s",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(pause)
    raise AssertionError("unreachable")


def download_artifacts(names: List[str], root: pathlib.Path) -> None:
    """Every unexpired artifact with one of `names`, from any run of the repository, into
    root/<run id>/<name>/ (one run per block or shard, retries included)."""
    repo = gh(
        ["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"]
    ).strip()
    for name in names:
        listing = gh(
            ["api", f"repos/{repo}/actions/artifacts?name={name}&per_page=100"]
        )
        for a in json.loads(listing).get("artifacts", []):
            if a.get("expired"):
                continue
            run = str(a["workflow_run"]["id"])
            target = root / run / name
            gh(["run", "download", run, "-n", name, "-D", str(target)], target)


def pick_attempt(dirs: List[pathlib.Path], complete, use: str | None = None):
    """The one attempt to join for a block or shard tried more than once: the run named with
    `use`, else the only complete attempt (a resumed attempt holds every game of the ones before
    it); None when that is ambiguous."""
    if use:
        named = [d for d in dirs if use in d.parts]
        return named[0] if len(named) == 1 else None
    done = [d for d in dirs if complete(d)]
    return done[0] if len(done) == 1 else (dirs[0] if len(dirs) == 1 else None)


def margins(d: pathlib.Path) -> set:
    """The fastchess time margins (ms) of a block attempt's invocations, from its manifest's
    history; an invocation that records none was played without one (before amendment 15 (j)).
    """
    try:
        history = json.load(open(d / "manifest.json")).get("history") or []
    except (OSError, ValueError):
        return set()
    return {int(h.get("timemargin_ms", 0)) for h in history}


def find_blocks(root: pathlib.Path, cell_id: str) -> Dict[int, List[pathlib.Path]]:
    found: Dict[int, List[pathlib.Path]] = {}
    for d in sorted(root.rglob("games-*")):
        m = BLOCK_DIR.match(d.name)
        if d.is_dir() and m and m.group("cell") == cell_id:
            found.setdefault(int(m.group("block")), []).append(d)
    return found


def join_cell(
    cell_id: str,
    cell: Dict,
    blocks: Dict[int, pathlib.Path],
    out: pathlib.Path,
    allow_gpu_errors: str | None = None,
    excluded: List[str] | None = None,
) -> Dict:
    """Write the joined cell directory from one complete directory per block; returns the
    summary (raises SystemExit on a missing, incomplete or inconsistent block, on a block played
    with another time margin than the cell's, and on GPU errors in the engines' stderr unless
    `allow_gpu_errors` names the amendment that treats them). `excluded` lists the attempts left
    out for their time margin, recorded in the manifest.
    """
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
    margin = int(cell.get("timemargin_ms", 0))
    for b in range(nblocks):
        if margins(blocks[b]) != {margin}:
            raise SystemExit(
                f"{cell_id} block {b}: played with time margins {sorted(margins(blocks[b]))} ms, "
                f"the cell's is {margin} ms (amendment 15 (j))"
            )
    gpu = {
        b: gpu_errors.count_in(blocks[b] / "engines.log.zst") for b in range(nblocks)
    }
    if any(gpu.values()) and not allow_gpu_errors:
        raise SystemExit(
            f"{cell_id}: the engines reported GPU errors on stderr (lines by block: {gpu}); "
            "the cell is read only after an amendment says how those games are treated "
            "(amendment 15 (i)), then with --allow-gpu-errors AMENDMENT"
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
    manifest.update(
        {
            "history": history,
            "runner_blocks": runners,
            "block": None,
            "gpu_error_lines_by_block": gpu,
            "gpu_errors_treated_by": allow_gpu_errors,
            "timemargin_ms": margin,
            "excluded_attempts": excluded or [],
        }
    )
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
    ap.add_argument(
        "--allow-gpu-errors",
        metavar="AMENDMENT",
        help="join a cell whose engines reported GPU errors, as the named amendment says",
    )
    args = ap.parse_args(argv)

    spec = json.load(open(args.manifest))
    cell = {c["id"]: c for c in spec["cells"]}.get(args.cell_id)
    if not cell or cell.get("machine") != "runner":
        raise SystemExit(f"{args.cell_id} is not a runner cell")
    nblocks = int(cell["blocks"])
    per_block = -(-int(cell["pairs"]) // nblocks)

    def complete(d: pathlib.Path) -> bool:
        b = int(BLOCK_DIR.match(d.name).group("block"))
        want = 2 * min(per_block, int(cell["pairs"]) - b * per_block)
        return count_games(d / "games.pgn") == want

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
        if not args.run and not args.from_dir:
            download_artifacts(
                [f"games-{args.cell_id}-b{b}" for b in range(nblocks)], root
            )
        found = find_blocks(root, args.cell_id)
        use = dict(u.split("=", 1) for u in args.use)
        margin = int(cell.get("timemargin_ms", 0))
        blocks, excluded = {}, []
        for b, dirs in found.items():
            # attempts played with another time margin (the runner's pilot blocks before
            # amendment 15 (j)) are never joined
            same = [d for d in dirs if margins(d) == {margin}]
            excluded += [str(d.relative_to(root)) for d in dirs if d not in same]
            if not same:
                continue
            pick = pick_attempt(same, complete, use.get(str(b)))
            if pick is None:
                raise SystemExit(
                    f"block {b}: {len(same)} attempts with the cell's time margin and no "
                    f"single complete one ({[str(d) for d in same]}); name the run with "
                    f"--use {b}=RUN"
                )
            blocks[b] = pick
        summary = join_cell(
            args.cell_id,
            cell,
            blocks,
            run_cell.RESULTS_ROOT / args.cell_id,
            args.allow_gpu_errors,
            sorted(excluded),
        )
    print(
        f"{args.cell_id}: {summary['stats'].get('games', '?')} games joined from "
        f"{len(blocks)} blocks into {run_cell.RESULTS_ROOT / args.cell_id}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
