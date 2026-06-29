#!/usr/bin/env python3
"""
MetalFish Parameter Tuning via SPRT

Systematically tests parameter values around a center point using SPRT,
narrowing in on the optimal value through binary search.

Usage:
    # Find optimal contempt between 300-800
    python3 tools/tune_parameter.py --param MCTSContempt --min 300 --max 800 --step 100

    # Fine-tune root CPUCT
    python3 tools/tune_parameter.py --param MCTSCPuctAtRoot --min 1.5 --max 2.2 --step 0.1

    # Quick test with fewer games (less accurate)
    python3 tools/tune_parameter.py --param MCTSContempt --min 400 --max 700 --step 100 --max-games 500 --movetime 500
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

PROJ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "tools"))
from sprt_test import (
    SPRTResult,
    default_hybrid_options,
    detect_threads,
    run_sprt,
)

RESULTS_DIR = PROJ / "results" / "tuning"


def format_value(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return f"{v:.4f}".rstrip("0").rstrip(".")


def tune_parameter(
    param: str,
    min_val: float,
    max_val: float,
    step: float,
    engine_cmd: str,
    base_options: dict,
    engine_cwd: str,
    elo0: float = -5.0,
    elo1: float = 5.0,
    max_games: int = 1000,
    movetime_ms: int = 1000,
    tc_base_ms: int = 0,
    tc_inc_ms: int = 0,
    verbose: bool = True,
) -> list:
    """Test parameter values from min to max in steps, return sorted results."""
    values = []
    v = min_val
    while v <= max_val + step / 2:
        values.append(round(v, 6))
        v += step

    results = []
    baseline_val = base_options.get(param, "?")

    print(f"\n{'='*60}")
    print(f"  Tuning: {param}")
    print(f"  Range: [{min_val}, {max_val}] step={step}")
    print(f"  Baseline value: {baseline_val}")
    print(f"  Testing {len(values)} values: {[format_value(v) for v in values]}")
    print(f"{'='*60}\n")

    for i, val in enumerate(values):
        val_str = format_value(val)
        if val_str == str(baseline_val):
            print(f"  [{i+1}/{len(values)}] {param}={val_str} (skip: same as baseline)")
            continue

        print(f"\n  [{i+1}/{len(values)}] Testing {param}={val_str}")

        candidate_options = dict(base_options)
        candidate_options[param] = val_str

        result = run_sprt(
            baseline_cmd=engine_cmd,
            candidate_cmd=engine_cmd,
            baseline_options=base_options,
            candidate_options=candidate_options,
            baseline_cwd=engine_cwd,
            candidate_cwd=engine_cwd,
            elo0=elo0,
            elo1=elo1,
            max_games=max_games,
            movetime_ms=movetime_ms,
            tc_base_ms=tc_base_ms,
            tc_inc_ms=tc_inc_ms,
            label=f"{param}={val_str}",
            verbose=verbose,
        )

        results.append(
            {
                "param": param,
                "value": val_str,
                "result": result.as_dict(),
            }
        )

        status_str = {"H1": "BETTER", "H0": "WORSE", "max_games": "NEUTRAL"}.get(
            result.status, result.status
        )
        print(
            f"  >> {status_str}: {param}={val_str} | "
            f"Elo={result.elo_est:+.1f} [{result.elo_ci_lo:+.1f}, {result.elo_ci_hi:+.1f}]"
        )

    results.sort(key=lambda r: r["result"]["elo_estimate"], reverse=True)

    print(f"\n{'='*60}")
    print(f"  Results for {param} (sorted by Elo):")
    print(f"{'='*60}")
    for r in results:
        elo = r["result"]["elo_estimate"]
        ci = r["result"]["elo_95ci"]
        status = r["result"]["status"]
        icon = {"H1": "+", "H0": "-", "max_games": "~"}.get(status, "?")
        print(
            f"  [{icon}] {param}={r['value']:>8s} | Elo={elo:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}] | {status}"
        )
    print()

    if results:
        best = results[0]
        print(
            f"  Best: {param}={best['value']} (Elo={best['result']['elo_estimate']:+.1f})"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="MetalFish Parameter Tuning via SPRT")
    parser.add_argument("--param", required=True, help="UCI option name to tune")
    parser.add_argument("--min", type=float, required=True, help="Minimum value")
    parser.add_argument("--max", type=float, required=True, help="Maximum value")
    parser.add_argument("--step", type=float, required=True, help="Step size")
    parser.add_argument("--engine", default=str(PROJ / "build" / "metalfish"))
    parser.add_argument(
        "--weights", default=str(PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb")
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--hash", type=int, default=2048)
    parser.add_argument("--mode", choices=["hybrid", "ab", "mcts"], default="hybrid")
    parser.add_argument("--elo0", type=float, default=-5.0)
    parser.add_argument("--elo1", type=float, default=5.0)
    parser.add_argument("--max-games", type=int, default=1000)
    parser.add_argument("--movetime", type=int, default=1000)
    parser.add_argument("--tc", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    threads = args.threads if args.threads > 0 else detect_threads()

    if args.mode == "hybrid":
        base_options = default_hybrid_options(args.weights, threads)
    elif args.mode == "ab":
        base_options = {
            "UseHybridSearch": "false",
            "UseMCTS": "false",
            "Threads": str(threads),
            "Hash": str(args.hash),
            "MultiPV": "1",
        }
    else:
        base_options = {
            "UseHybridSearch": "false",
            "UseMCTS": "true",
            "NNWeights": args.weights,
            "Threads": str(threads),
            "Hash": str(args.hash),
            "MCTSMaxThreads": "1",
            "MCTSMinibatchSize": "0",
            "MCTSParityPreset": "false",
            "MCTSAddDirichletNoise": "false",
            "TransformerLowTimeFallbackMs": "0",
        }

    base_options["Hash"] = str(args.hash)

    tc_base_ms = 0
    tc_inc_ms = 0
    if args.tc:
        parts = args.tc.replace("+", " ").split()
        tc_base_ms = int(float(parts[0]) * 1000)
        tc_inc_ms = int(float(parts[1]) * 1000) if len(parts) > 1 else 0

    results = tune_parameter(
        param=args.param,
        min_val=args.min,
        max_val=args.max,
        step=args.step,
        engine_cmd=args.engine,
        base_options=base_options,
        engine_cwd=str(PROJ),
        elo0=args.elo0,
        elo1=args.elo1,
        max_games=args.max_games,
        movetime_ms=args.movetime,
        tc_base_ms=tc_base_ms,
        tc_inc_ms=tc_inc_ms,
        verbose=not args.quiet,
    )

    out_path = args.json_out
    if not out_path:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = str(RESULTS_DIR / f"tune_{args.param}_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
