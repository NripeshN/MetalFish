#!/usr/bin/env python3
"""Join the RQ6b-R oracle shards into results/paper_v2/RQ6b-R/ (protocol amendment 15 (a)).

Shard i of K is one job of .github/workflows/paper-runner-cells.yml (task oracle) and leaves an
artifact oracle-shard-<i> with shard_<i>.jsonl (one row per judged position, in sample order) and the
shard's runner record, health samples and filtered stderr. This tool downloads every attempt (the
artifacts API, any run), takes for each shard its one complete attempt (amendment 15 (h); a resumed
attempt holds the rows of the attempts before it), refuses a missing shard, and writes the rows in
shard order -- the order of a single run -- to oracle.jsonl, with each shard's other files under
shards/<i>/. The reading is then one `oracle_probe.py --summarise-only` over oracle.jsonl with the
registered sample arguments.

An attempt is complete when every position of the shard has a usable judgment from both judges:
the test the shard's own job applies before it succeeds (amendment 15 (i) 2). A row is usable when
it holds both judges, neither with an error, and an Lc0 judgment that records its configuration,
so none made on a runner's GPU, all of which are discarded (15 (i) 1, (k) 2). The attempts of
2026-11-04 to 06 that the GPU failure left with all their rows but Lc0 errors, or Lc0 judgments
without a configuration, are therefore not complete; the attempt that resumed them is.

With --stability WORKSTATION_ROWS it also writes stability.json: for the positions the workstation
judged before it was stopped, judge by judge, the share of equal verdicts and the correlation of the
candidates' score differences on the two machines (a check of judge stability; it enters no
estimate).

    python3 tools/paper/fetch_runner_oracle.py --shards 150 [--stability results/paper_v2/RQ6b/oracle.jsonl]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import statistics
import sys
import tempfile
from typing import Dict, List

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "paper"))

import fetch_runner_cell  # noqa: E402

SHARD_DIR = re.compile(r"^oracle-shard-(\d+)$")
# the oracle's two judges, named as the workflow's oracle job names them
JUDGES = {"sf19", "lc0"}


def shard_size(i: int, k: int, sample: int) -> int:
    return (i + 1) * sample // k - i * sample // k


def rows_of(path: pathlib.Path) -> List[Dict]:
    return [json.loads(l) for l in open(path) if l.strip()] if path.exists() else []


def usable(row: Dict) -> bool:
    """Both judges, neither with an error, and an Lc0 judgment that records its configuration
    (the conditions of the oracle job's health step; amendment 15 (i) 1-2, (k) 2)."""
    judges = row.get("judges", {})
    return (
        set(judges) == JUDGES
        and not any("error" in j for j in judges.values())
        and "config" in judges["lc0"]
    )


def join_shards(
    found: Dict[int, List[pathlib.Path]], k: int, sample: int, out: pathlib.Path
) -> Dict:
    def complete(d: pathlib.Path) -> bool:
        i = int(SHARD_DIR.match(d.name).group(1))
        rows = rows_of(d / f"shard_{i}.jsonl")
        return len(rows) == shard_size(i, k, sample) and all(usable(r) for r in rows)

    missing = [i for i in range(k) if i not in found]
    if missing:
        raise SystemExit(f"shards {missing} have no artifact")
    picked = {}
    for i in range(k):
        d = fetch_runner_cell.pick_attempt(found[i], complete)
        if d is None or not complete(d):
            raise SystemExit(
                f"shard {i}: no single complete attempt among {[str(x) for x in found[i]]}"
            )
        picked[i] = d
    out.mkdir(parents=True, exist_ok=True)
    if (out / "oracle.jsonl").exists():
        raise SystemExit(f"{out / 'oracle.jsonl'} exists; the shards are joined once")
    shards = []
    errors = 0
    with open(out / "oracle.jsonl", "w") as fh:
        for i in range(k):
            d = picked[i]
            for row in rows_of(d / f"shard_{i}.jsonl"):
                errors += sum(1 for j in row["judges"].values() if "error" in j)
                fh.write(json.dumps(row) + "\n")
            dest = out / "shards" / str(i)
            dest.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.name != f"shard_{i}.jsonl":
                    shutil.copy2(f, dest / f.name)
            rj = d / f"runner_s{i}.json"
            runner = json.load(open(rj)) if rj.exists() else None
            shards.append(
                {
                    "shard": i,
                    "attempts": len(found[i]),
                    "run": d.parent.name,
                    "runner": runner,
                }
            )
    info = {
        "shards": k,
        "sample": sample,
        "positions": sum(shard_size(i, k, sample) for i in range(k)),
        "judge_errors": errors,
        "by_shard": shards,
    }
    json.dump(info, open(out / "shards.json", "w"), indent=1)
    return info


def judge_stability(runner_rows: List[Dict], workstation_rows: List[Dict]) -> Dict:
    """Per judge, over the positions judged on both machines: equal verdicts and the Pearson
    correlation of delta_cp (alpha-beta candidate minus MCTS candidate)."""

    def key(r):
        return (r["fen"], r["ab_move"], r["mcts_move"], r.get("cluster"))

    ws = {key(r): r for r in workstation_rows}
    out = {}
    for name in sorted({n for r in runner_rows for n in r["judges"]}):
        pairs = []
        for r in runner_rows:
            w = ws.get(key(r))
            a, b = r["judges"].get(name, {}), (w or {}).get("judges", {}).get(name, {})
            if w and "verdict" in a and "verdict" in b:
                pairs.append((a, b))
        if not pairs:
            out[name] = {"positions": 0}
            continue
        x = [a["delta_cp"] for a, _ in pairs]
        y = [b["delta_cp"] for _, b in pairs]
        corr = (
            statistics.correlation(x, y)
            if len(pairs) > 2 and len(set(x)) > 1 and len(set(y)) > 1
            else None
        )
        out[name] = {
            "positions": len(pairs),
            "same_verdict": sum(a["verdict"] == b["verdict"] for a, b in pairs)
            / len(pairs),
            "delta_cp_correlation": corr,
        }
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--shards", type=int, default=150)
    ap.add_argument("--sample", type=int, default=600)
    ap.add_argument(
        "--from-dir", help="directory holding downloaded oracle-shard-<i> artifacts"
    )
    ap.add_argument("--out", default=str(REPO / "results" / "paper_v2" / "RQ6b-R"))
    ap.add_argument(
        "--stability",
        help="the workstation's rows (results/paper_v2/RQ6b/oracle.jsonl)",
    )
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(args.from_dir or tmp)
        if not args.from_dir:
            fetch_runner_cell.download_artifacts(
                [f"oracle-shard-{i}" for i in range(args.shards)], root
            )
        found: Dict[int, List[pathlib.Path]] = {}
        for d in sorted(root.rglob("oracle-shard-*")):
            m = SHARD_DIR.match(d.name)
            if d.is_dir() and m:
                found.setdefault(int(m.group(1)), []).append(d)
        info = join_shards(found, args.shards, args.sample, pathlib.Path(args.out))
    print(
        f"{info['positions']} positions from {info['shards']} shards; judge errors {info['judge_errors']}"
    )
    if args.stability:
        st = judge_stability(
            rows_of(pathlib.Path(args.out) / "oracle.jsonl"),
            rows_of(pathlib.Path(args.stability)),
        )
        json.dump(st, open(pathlib.Path(args.out) / "stability.json", "w"), indent=1)
        print(json.dumps(st, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
