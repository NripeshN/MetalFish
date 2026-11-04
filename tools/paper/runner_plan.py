#!/usr/bin/env python3
"""Job matrix of a paper runner campaign (.github/workflows/paper-runner-cells.yml, protocol
amendment 15).

games:  one job per block of each runner cell named in CELLS, interleaved block by block
        (block 0 of every cell, then block 1, ...), so that the cells of a contrast meet the
        same mix of runners and images; BLOCKS optionally restricts the blocks (a re-run).
oracle: one job per contiguous shard of the RQ6b sample; SHARD_LIST optionally restricts them.

Inputs come from the environment (TASK, CELLS, BLOCKS, SHARDS, SHARD_LIST); prints
`matrix=<json>` for $GITHUB_OUTPUT and the plan in words on stderr.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Dict, List, Optional

REPO = pathlib.Path(__file__).resolve().parents[2]


def int_list(text: str) -> Optional[List[int]]:
    text = (text or "").strip()
    return [int(x) for x in text.split(",") if x.strip()] if text else None


def games_matrix(spec: Dict, cells: List[str], only: Optional[List[int]]) -> List[Dict]:
    by_id = {c["id"]: c for c in spec["cells"]}
    nblocks = {}
    for cid in cells:
        cell = by_id.get(cid)
        if cell is None:
            raise SystemExit(f"unknown cell {cid}")
        if cell.get("machine") != "runner":
            raise SystemExit(f"{cid} is not a runner cell (amendment 15)")
        nblocks[cid] = int(cell["blocks"])
    jobs = []
    for b in range(max(nblocks.values(), default=0)):
        for cid in cells:
            if b < nblocks[cid] and (only is None or b in only):
                jobs.append({"cell": cid, "block": b})
    return jobs


def oracle_matrix(shards: int, only: Optional[List[int]]) -> List[Dict]:
    return [{"shard": i} for i in range(shards) if only is None or i in only]


def main() -> int:
    task = os.environ.get("TASK", "")
    if task == "games":
        spec = json.load(open(REPO / "docs" / "paper_v2" / "experiments.json"))
        cells = [c.strip() for c in os.environ.get("CELLS", "").split(",") if c.strip()]
        jobs = games_matrix(spec, cells, int_list(os.environ.get("BLOCKS", "")))
        words = [f"{j['cell']} block {j['block']}" for j in jobs]
    elif task == "oracle":
        shards = int(os.environ.get("SHARDS", "0"))
        jobs = oracle_matrix(shards, int_list(os.environ.get("SHARD_LIST", "")))
        words = [f"shard {j['shard']}/{shards}" for j in jobs]
    else:
        raise SystemExit(f"no matrix for task {task!r}")
    if not jobs:
        raise SystemExit("empty plan")
    print(f"{len(jobs)} jobs: " + ", ".join(words), file=sys.stderr)
    print("matrix=" + json.dumps({"include": jobs}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
