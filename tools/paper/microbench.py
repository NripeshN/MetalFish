#!/usr/bin/env python3
"""Microbenchmarks for the systems section (M1 NN latency sweep, M2 AB thread scaling).

M1: python3 tools/paper/microbench.py nn --out results/paper_v2/micro/nn_latency.json
    Runs build/metalfish_nn_probe (Metal, FP16 and FP32 via METALFISH_METAL_FP16) over batch sizes and,
    if lc0 is available, `lc0 benchmark` over minibatch sizes with the same weights.
M2: python3 tools/paper/microbench.py ab --out results/paper_v2/micro/ab_scaling.json
    Runs `metalfish bench <hash> <threads> <depth>` for several thread counts, N repeats each; reports
    nodes/s and wall time. Depth-limited so the node count is fixed per thread count (reported too).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Optional

REPO = pathlib.Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "networks" / "BT4-1024x15x32h-swa-6147500.pb"


def run(
    cmd: List[str], env: Optional[Dict[str, str]] = None, timeout: int = 1800
) -> str:
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **(env or {})},
    )
    return p.stdout + p.stderr


def nn_sweep(batches: List[int], iterations: int, lc0: Optional[str]) -> Dict:
    out: Dict = {"metalfish": [], "lc0": []}
    probe = REPO / "build" / "metalfish_nn_probe"
    for fp16 in (1, 0):
        for b in batches:
            txt = run(
                [
                    str(probe),
                    "--weights",
                    str(WEIGHTS),
                    "--backend",
                    "metal",
                    "--batch-size",
                    str(b),
                    "--warmup",
                    "5",
                    "--iterations",
                    str(iterations),
                ],
                env={"METALFISH_METAL_FP16": str(fp16)},
            )
            m = re.search(r'"latency":(\{[^}]*\})', txt)
            lat = json.loads(m.group(1)) if m else {"error": txt[-300:]}
            out["metalfish"].append(
                {"precision": "fp16" if fp16 else "fp32", "batch": b, **lat}
            )
            print(
                f"metalfish fp16={fp16} batch={b}: {lat.get('median_ms')} ms, {lat.get('median_positions_per_second')} pos/s"
            )
    if lc0:
        for mb in batches:
            txt = run(
                [
                    lc0,
                    "benchmark",
                    f"--weights={WEIGHTS}",
                    "--backend=metal",
                    f"--minibatch-size={mb}",
                    "--threads=2",
                    "--movetime=15000",
                    "--num-positions=3",
                ],
                timeout=600,
            )
            nps = [int(x) for x in re.findall(r"(\d+) nps", txt)]
            out["lc0"].append(
                {
                    "minibatch": mb,
                    "nps_per_position": nps,
                    # Median over all per-second samples (includes the tree-growth
                    # ramp-up, so it is biased low) and the steady-state rate
                    # (median of the last five samples), which is the one to report.
                    "nps_median": statistics.median(nps) if nps else None,
                    "nps_steady_median": (statistics.median(nps[-5:]) if nps else None),
                    "nps_max": max(nps) if nps else None,
                }
            )
            print(f"lc0 minibatch={mb}: {nps}")
    return out


def ab_scaling(
    threads: List[int], repeats: int, depth: int, hash_mb: int, engine: str
) -> Dict:
    rows = []
    for t in threads:
        for r in range(repeats):
            t0 = time.time()
            txt = run([engine, "bench", str(hash_mb), str(t), str(depth)], timeout=3600)
            el = time.time() - t0
            nodes = re.search(r"Nodes searched\s*:\s*(\d+)", txt)
            nps = re.search(r"Nodes/second\s*:\s*(\d+)", txt)
            rows.append(
                {
                    "threads": t,
                    "repeat": r,
                    "nodes": int(nodes.group(1)) if nodes else None,
                    "nps": int(nps.group(1)) if nps else None,
                    "wall_s": el,
                }
            )
            print(rows[-1])
    agg = {}
    for t in threads:
        xs = [x["nps"] for x in rows if x["threads"] == t and x["nps"]]
        agg[t] = {
            "nps_median": statistics.median(xs) if xs else None,
            "nps_min": min(xs) if xs else None,
            "nps_max": max(xs) if xs else None,
        }
    base = (
        agg[threads[0]]["nps_median"]
        if agg.get(threads[0], {}).get("nps_median")
        else None
    )
    for t in threads:
        agg[t]["speedup_vs_first"] = (
            (agg[t]["nps_median"] / base) if base and agg[t]["nps_median"] else None
        )
    return {
        "engine": engine,
        "depth": depth,
        "hash_mb": hash_mb,
        "rows": rows,
        "aggregate": agg,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("which", choices=["nn", "ab"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", default="1,2,4,8,16,32,64")
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--lc0", default="/opt/homebrew/bin/lc0")
    ap.add_argument("--threads", default="1,2,4,6,8,12")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--hash", type=int, default=1024)
    ap.add_argument("--engine", default=str(REPO / "build" / "metalfish"))
    args = ap.parse_args(argv)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.which == "nn":
        res = nn_sweep(
            [int(x) for x in args.batches.split(",")],
            args.iterations,
            args.lc0 if pathlib.Path(args.lc0).exists() else None,
        )
    else:
        res = ab_scaling(
            [int(x) for x in args.threads.split(",")],
            args.repeats,
            args.depth,
            args.hash,
            args.engine,
        )
    res["hardware"] = run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    # a measurement the engine or Lc0 failed to produce is written for diagnosis but fails the run,
    # so the queue log records a non-zero exit instead of a sweep full of nulls
    if args.which == "nn":
        failed = [r for r in res["metalfish"] if r.get("median_ms") is None]
        failed += [r for r in res["lc0"] if r.get("nps_steady_median") is None]
    else:
        failed = [r for r in res["rows"] if r.get("nps") is None]
    if failed:
        print(f"{len(failed)} measurement(s) failed; see {args.out}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
