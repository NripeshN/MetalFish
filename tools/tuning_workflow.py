#!/usr/bin/env python3
"""MetalFish tuning workflow: rank candidate configs against a fixed baseline.

This orchestrates :mod:`tools.sprt_test` to answer one question repeatably:

    "Does candidate configuration X make the engine stronger than the current
     (baseline) engine, and by how much?"

The *baseline* is always the current engine build + the mode's default options
(the same configuration the Lichess bot plays with for ``hybrid``). Each
*candidate* is the baseline with a small set of UCI option overrides (or a
different binary via ``--candidate-engine``). Every candidate plays a colour-
balanced match against the baseline; results are ranked by Elo and the best
statistically-validated candidate is recommended as the new default.

Typical use
-----------
Fast search-parameter screening (Alpha-Beta, thousands of games/hour)::

    python3 tools/tuning_workflow.py --mode ab --movetime 100 \
        --candidates tools/tuning_candidates.json --max-games 400

Fighting-engine draw-aversion sweep in the hybrid engine the bot actually
plays (slower; run overnight or reduce --max-games)::

    python3 tools/tuning_workflow.py --mode hybrid --movetime 800 \
        --candidates tools/tuning_candidates.json --max-games 200

See ``docs/tuning_workflow.md`` for the end-to-end methodology.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

PROJ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from tools import sprt_test  # noqa: E402

RESULTS_DIR = PROJ / "results" / "tuning"
DEFAULT_CANDIDATES = PROJ / "tools" / "tuning_candidates.json"


# ---------------------------------------------------------------------------
# Candidate configuration (pure, unit-testable)
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    label: str
    options: dict[str, str]
    elo0: float = 0.0
    elo1: float = 5.0
    note: str = ""


def load_candidate_config(
    path: str | pathlib.Path,
) -> tuple[dict[str, str], list[Candidate]]:
    """Parse a candidate config file into (baseline_overrides, candidates).

    Schema::

        {
          "baseline_options": {"UCIName": "value", ...},   # optional
          "elo0": 0.0, "elo1": 5.0,                          # optional defaults
          "candidates": [
            {"label": "contempt-300", "options": {"MCTSContempt": "300"},
             "elo0": 0, "elo1": 6, "note": "..."},
            ...
          ]
        }
    """
    with open(path) as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("candidate config must be a JSON object")

    baseline_overrides = {
        str(k): str(v) for k, v in (data.get("baseline_options") or {}).items()
    }
    default_elo0 = float(data.get("elo0", 0.0))
    default_elo1 = float(data.get("elo1", 5.0))

    raw_candidates = data.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidate config must contain a non-empty 'candidates' list")

    candidates: list[Candidate] = []
    seen: set[str] = set()
    for item in raw_candidates:
        if not isinstance(item, dict):
            raise ValueError("each candidate must be a JSON object")
        options = item.get("options")
        if not isinstance(options, dict) or not options:
            raise ValueError(
                f"candidate {item.get('label')!r} needs a non-empty 'options'"
            )
        label = str(
            item.get("label") or "+".join(f"{k}={v}" for k, v in options.items())
        )
        if label in seen:
            raise ValueError(f"duplicate candidate label: {label}")
        seen.add(label)
        candidates.append(
            Candidate(
                label=label,
                options={str(k): str(v) for k, v in options.items()},
                elo0=float(item.get("elo0", default_elo0)),
                elo1=float(item.get("elo1", default_elo1)),
                note=str(item.get("note", "")),
            )
        )
    return baseline_overrides, candidates


def base_options_for_mode(
    mode: str, weights: str, threads: int, hash_mb: int
) -> dict[str, str]:
    """Baseline UCI options for a mode, mirroring tools/sprt_test.py."""
    if mode == "hybrid":
        options = sprt_test.default_hybrid_options(weights, threads)
    elif mode == "ab":
        options = {
            "UseHybridSearch": "false",
            "UseMCTS": "false",
            "Threads": str(threads),
            "MultiPV": "1",
        }
    elif mode == "mcts":
        options = {
            "UseHybridSearch": "false",
            "UseMCTS": "true",
            "NNWeights": weights,
            "Threads": str(threads),
            "MCTSMaxThreads": "1",
            "MCTSMinibatchSize": "0",
            "MCTSParityPreset": "false",
            "MCTSAddDirichletNoise": "false",
            "TransformerLowTimeFallbackMs": "0",
        }
    else:
        raise ValueError(f"unknown mode: {mode}")
    options["Hash"] = str(hash_mb)
    return options


# ---------------------------------------------------------------------------
# Result ranking / recommendation (pure, unit-testable)
# ---------------------------------------------------------------------------


@dataclass
class CandidateOutcome:
    label: str
    options: dict[str, str]
    wins: int = 0
    draws: int = 0
    losses: int = 0
    elo: float = 0.0
    elo_lo: float = 0.0
    elo_hi: float = 0.0
    llr: float = 0.0
    status: str = ""
    games: int = 0
    elapsed_sec: float = 0.0
    note: str = ""

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def draw_ratio(self) -> float:
        return self.draws / self.total if self.total else 0.0

    @property
    def score_pct(self) -> float:
        if not self.total:
            return 0.0
        return 100.0 * (self.wins + 0.5 * self.draws) / self.total

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "options": self.options,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "games": self.total,
            "score_pct": round(self.score_pct, 2),
            "draw_ratio": round(self.draw_ratio, 3),
            "elo": round(self.elo, 1),
            "elo_95ci": [round(self.elo_lo, 1), round(self.elo_hi, 1)],
            "llr": round(self.llr, 3),
            "status": self.status,
            "elapsed_sec": round(self.elapsed_sec, 1),
            "note": self.note,
        }


def outcome_from_sprt(
    candidate: Candidate, result: "sprt_test.SPRTResult"
) -> CandidateOutcome:
    return CandidateOutcome(
        label=candidate.label,
        options=candidate.options,
        wins=result.wins,
        draws=result.draws,
        losses=result.losses,
        elo=result.elo_est,
        elo_lo=result.elo_ci_lo,
        elo_hi=result.elo_ci_hi,
        llr=result.llr,
        status=result.status,
        games=result.total,
        elapsed_sec=result.elapsed_sec,
        note=candidate.note,
    )


def rank_outcomes(outcomes: list[CandidateOutcome]) -> list[CandidateOutcome]:
    """Best first: SPRT-pass, then higher Elo, then tighter/positive lower CI."""

    def key(o: CandidateOutcome):
        passed = 1 if o.status == "H1" else 0
        rejected = 1 if o.status == "H0" else 0
        return (passed, -rejected, o.elo, o.elo_lo)

    return sorted(outcomes, key=key, reverse=True)


def recommend(
    outcomes: list[CandidateOutcome], min_games: int = 40
) -> Optional[CandidateOutcome]:
    """Recommend the strongest candidate that is *statistically* better.

    A candidate qualifies only if it either passed SPRT (H1) or its 95%%
    lower Elo bound is above zero over a meaningful sample. Returns ``None``
    when no candidate clears the bar (i.e. keep the current default).
    """
    qualified = [
        o
        for o in outcomes
        if o.total >= min_games
        and o.status != "H0"
        and (o.status == "H1" or o.elo_lo > 0.0)
    ]
    if not qualified:
        return None
    return rank_outcomes(qualified)[0]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_workflow(
    *,
    baseline_engine: str,
    candidate_engine: str,
    base_options: dict[str, str],
    candidates: list[Candidate],
    movetime_ms: int,
    tc_base_ms: int,
    tc_inc_ms: int,
    max_games: int,
    alpha: float,
    beta: float,
    verbose: bool,
) -> list[CandidateOutcome]:
    outcomes: list[CandidateOutcome] = []
    for idx, candidate in enumerate(candidates, 1):
        candidate_options = dict(base_options)
        candidate_options.update(candidate.options)
        overrides = ", ".join(f"{k}={v}" for k, v in candidate.options.items())
        print(f"\n[{idx}/{len(candidates)}] {candidate.label}  ({overrides})")
        if candidate.note:
            print(f"    note: {candidate.note}")
        print(
            f"    H0: Elo<={candidate.elo0}  H1: Elo>={candidate.elo1}  "
            f"max_games={max_games}"
        )
        result = sprt_test.run_sprt(
            baseline_cmd=baseline_engine,
            candidate_cmd=candidate_engine,
            baseline_options=base_options,
            candidate_options=candidate_options,
            baseline_cwd=str(PROJ),
            candidate_cwd=str(PROJ),
            elo0=candidate.elo0,
            elo1=candidate.elo1,
            alpha=alpha,
            beta=beta,
            max_games=max_games,
            movetime_ms=movetime_ms,
            tc_base_ms=tc_base_ms,
            tc_inc_ms=tc_inc_ms,
            label=candidate.label,
            verbose=verbose,
        )
        outcome = outcome_from_sprt(candidate, result)
        outcomes.append(outcome)
        print(
            f"    => {outcome.status}: Elo {outcome.elo:+.1f} "
            f"[{outcome.elo_lo:+.1f}, {outcome.elo_hi:+.1f}] "
            f"W{outcome.wins} D{outcome.draws} L{outcome.losses} "
            f"draw%={outcome.draw_ratio*100:.0f} ({outcome.total} games)"
        )
    return outcomes


def format_report(
    outcomes: list[CandidateOutcome], recommended: Optional[CandidateOutcome]
) -> str:
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("  TUNING WORKFLOW SUMMARY (candidate vs current baseline)")
    lines.append("=" * 78)
    header = (
        f"  {'candidate':<22}{'Elo (95% CI)':<22}{'W-D-L':<14}{'draw%':<7}{'status'}"
    )
    lines.append(header)
    lines.append("  " + "-" * 74)
    for o in rank_outcomes(outcomes):
        elo_str = f"{o.elo:+.1f} [{o.elo_lo:+.1f},{o.elo_hi:+.1f}]"
        wdl = f"{o.wins}-{o.draws}-{o.losses}"
        lines.append(
            f"  {o.label:<22}{elo_str:<22}{wdl:<14}{o.draw_ratio*100:<7.0f}{o.status}"
        )
    lines.append("  " + "-" * 74)
    if recommended is not None:
        lines.append(
            f"  RECOMMENDED NEW DEFAULT: {recommended.label} "
            f"(Elo {recommended.elo:+.1f}, lower bound {recommended.elo_lo:+.1f})"
        )
        lines.append(f"    options: {recommended.options}")
    else:
        lines.append(
            "  RECOMMENDATION: keep current default (no candidate proved stronger)"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank candidate engine configs against the current baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--engine",
        default=str(PROJ / "build" / "metalfish"),
        help="Baseline engine binary (the current engine).",
    )
    parser.add_argument(
        "--candidate-engine",
        default=None,
        help="Candidate binary (default: same as --engine; use for code changes).",
    )
    parser.add_argument(
        "--weights", default=str(PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb")
    )
    parser.add_argument("--mode", choices=["hybrid", "ab", "mcts"], default="hybrid")
    parser.add_argument("--threads", type=int, default=0, help="0 = auto-detect")
    parser.add_argument("--hash", type=int, default=2048)
    parser.add_argument(
        "--candidates",
        default=str(DEFAULT_CANDIDATES),
        help="Candidate config JSON (see docs/tuning_workflow.md).",
    )
    parser.add_argument(
        "--movetime",
        type=int,
        default=0,
        help="Fixed ms/move (used if --tc unset). Default picks a "
        "mode-appropriate value.",
    )
    parser.add_argument(
        "--tc", default=None, help="Time control BASE+INC in seconds, e.g. '10+0.1'."
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=400,
        help="Games per candidate before declaring inconclusive.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    threads = args.threads if args.threads > 0 else sprt_test.detect_threads()
    baseline_engine = args.engine
    candidate_engine = args.candidate_engine or args.engine

    tc_base_ms = tc_inc_ms = 0
    if args.tc:
        parts = args.tc.replace("+", " ").split()
        tc_base_ms = int(float(parts[0]) * 1000)
        tc_inc_ms = int(float(parts[1]) * 1000) if len(parts) > 1 else 0
        movetime_ms = 0
    else:
        # Mode-appropriate default movetime: AB is cheap, hybrid/mcts need more.
        movetime_ms = args.movetime or (100 if args.mode == "ab" else 800)

    base_options = base_options_for_mode(args.mode, args.weights, threads, args.hash)
    baseline_overrides, candidates = load_candidate_config(args.candidates)
    base_options.update(baseline_overrides)

    print("=" * 78)
    print("  MetalFish Tuning Workflow")
    print(f"  Baseline:   {baseline_engine}")
    if candidate_engine != baseline_engine:
        print(f"  Candidate:  {candidate_engine}")
    print(f"  Mode: {args.mode} | Threads: {threads} | Hash: {args.hash}MB")
    if tc_base_ms:
        print(f"  TC: {tc_base_ms/1000:.1f}+{tc_inc_ms/1000:.2f}s")
    else:
        print(f"  Movetime: {movetime_ms}ms")
    print(f"  Candidates: {len(candidates)} from {args.candidates}")
    print(f"  Baseline overrides: {baseline_overrides or 'none'}")
    print("=" * 78)

    start = time.time()
    outcomes = run_workflow(
        baseline_engine=baseline_engine,
        candidate_engine=candidate_engine,
        base_options=base_options,
        candidates=candidates,
        movetime_ms=movetime_ms,
        tc_base_ms=tc_base_ms,
        tc_inc_ms=tc_inc_ms,
        max_games=args.max_games,
        alpha=args.alpha,
        beta=args.beta,
        verbose=not args.quiet,
    )
    recommended = recommend(outcomes)
    print(format_report(outcomes, recommended))

    out_path = args.json_out
    if not out_path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = str(RESULTS_DIR / f"tuning_{args.mode}_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "mode": args.mode,
        "baseline_engine": baseline_engine,
        "candidate_engine": candidate_engine,
        "movetime_ms": movetime_ms,
        "tc_base_ms": tc_base_ms,
        "tc_inc_ms": tc_inc_ms,
        "max_games": args.max_games,
        "elapsed_sec": round(time.time() - start, 1),
        "baseline_overrides": baseline_overrides,
        "results": [o.as_dict() for o in rank_outcomes(outcomes)],
        "recommended": recommended.label if recommended else None,
    }
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nResults saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
