#!/usr/bin/env python3
"""Compare two offline/online puzzle-run JSONL reports."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def load_results(path: pathlib.Path) -> list[dict]:
    results: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            puzzle_id = str(item.get("id", ""))
            if puzzle_id:
                results.append(item)
    if not results:
        raise ValueError(f"{path}: no puzzle results found")
    return results


def result_id(item: dict) -> str:
    return str(item.get("id", ""))


def base_result_id(item: dict) -> str:
    puzzle_id = str(item.get("puzzle_id", ""))
    if puzzle_id:
        return puzzle_id
    item_id = result_id(item)
    if "#r" in item_id:
        return item_id.split("#r", 1)[0]
    return item_id


def unique_by_id(results: list[dict]) -> dict[str, dict]:
    keyed: dict[str, dict] = {}
    for item in results:
        item_id = result_id(item)
        if item_id:
            keyed[item_id] = item
    return keyed


def pair_results(
    baseline: list[dict], candidate: list[dict], *, match_repeat_ids: bool
) -> list[tuple[str, dict, dict]]:
    if not match_repeat_ids:
        baseline_by_id = unique_by_id(baseline)
        candidate_by_id = unique_by_id(candidate)
        return [
            (item_id, baseline_by_id[item_id], candidate_by_id[item_id])
            for item_id in sorted(set(baseline_by_id).intersection(candidate_by_id))
        ]

    baseline_by_id = unique_by_id(baseline)
    baseline_by_base: dict[str, list[dict]] = {}
    for item in baseline:
        base_id = base_result_id(item)
        if base_id:
            baseline_by_base.setdefault(base_id, []).append(item)

    pairs: list[tuple[str, dict, dict]] = []
    for item in candidate:
        item_id = result_id(item)
        if item_id in baseline_by_id:
            pairs.append((item_id, baseline_by_id[item_id], item))
            continue
        base_id = base_result_id(item)
        base_matches = baseline_by_base.get(base_id, [])
        if len(base_matches) == 1:
            pairs.append((item_id or base_id, base_matches[0], item))
    return pairs


def solved_count(results: list[dict]) -> int:
    return sum(1 for item in results if item.get("solved"))


def error_count(results: list[dict]) -> int:
    return sum(1 for item in results if item.get("error"))


def iter_search_records(results: list[dict] | dict[str, dict]):
    items = results.values() if isinstance(results, dict) else results
    for item in items:
        puzzle_id = result_id(item)
        searches = item.get("searches", [])
        if not isinstance(searches, list):
            continue
        for search in searches:
            if isinstance(search, dict):
                yield puzzle_id, bool(item.get("solved")), search


def first_move(move_list: str | None) -> str:
    if not isinstance(move_list, str):
        return ""
    return move_list.split()[0] if move_list.split() else ""


def parse_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ane_trace_summary(results: list[dict] | dict[str, dict]) -> dict[str, object]:
    reason_counts: dict[str, int] = {}
    blocked_examples: list[str] = []
    total = 0
    ane_hint_searches = 0
    ane_mcts_agree = 0
    ane_mcts_selected = 0
    ane_mcts_blocked_by_ab = 0
    ane_confirmed_mcts = 0
    unsolved_blocked = 0
    score_margins: list[float] = []

    for puzzle_id, solved, search in iter_search_records(results):
        total += 1
        ane_top = str(search.get("hybrid_ane_top") or "") or first_move(
            search.get("ane_last_hints")
        )
        mcts_move = str(search.get("hybrid_mcts_move") or "")
        ab_move = str(search.get("hybrid_ab_move") or "")
        selected = str(search.get("hybrid_selected") or search.get("actual") or "")
        reason = str(search.get("hybrid_reason") or "")
        margin = parse_float(search.get("hybrid_ane_score_margin"))
        if margin is not None:
            score_margins.append(margin)
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if (
            reason == "ane_confirmed_mcts"
            or str(search.get("hybrid_ane_confirmed_mcts") or "") == "1"
        ):
            ane_confirmed_mcts += 1
        if not ane_top:
            continue
        ane_hint_searches += 1
        if ane_top != mcts_move:
            continue
        ane_mcts_agree += 1
        if selected == ane_top:
            ane_mcts_selected += 1
            continue
        if selected == ab_move and ab_move:
            ane_mcts_blocked_by_ab += 1
            if not solved:
                unsolved_blocked += 1
            if len(blocked_examples) < 8:
                ply = search.get("ply", "?")
                expected = search.get("expected", "?")
                blocked_examples.append(
                    f"{puzzle_id}@ply{ply}: ane/mcts={ane_top}, "
                    f"selected/ab={selected}, expected={expected}, reason={reason}"
                )

    sorted_margins = sorted(score_margins)
    return {
        "searches": total,
        "ane_hint_searches": ane_hint_searches,
        "ane_mcts_agree": ane_mcts_agree,
        "ane_mcts_selected": ane_mcts_selected,
        "ane_mcts_blocked_by_ab": ane_mcts_blocked_by_ab,
        "ane_confirmed_mcts": ane_confirmed_mcts,
        "unsolved_blocked": unsolved_blocked,
        "reason_counts": reason_counts,
        "blocked_examples": blocked_examples,
        "ane_score_margin_min": sorted_margins[0] if sorted_margins else None,
        "ane_score_margin_median": (
            sorted_margins[len(sorted_margins) // 2] if sorted_margins else None
        ),
        "ane_score_margin_max": sorted_margins[-1] if sorted_margins else None,
    }


def print_ane_trace_summary(results: list[dict] | dict[str, dict], label: str) -> None:
    summary = ane_trace_summary(results)
    print(
        f"ANE trace {label}: searches={summary['searches']}, "
        f"ane_hint_searches={summary['ane_hint_searches']}, "
        f"ane_mcts_agree={summary['ane_mcts_agree']}, "
        f"ane_mcts_selected={summary['ane_mcts_selected']}, "
        f"ane_mcts_blocked_by_ab={summary['ane_mcts_blocked_by_ab']}, "
        f"ane_confirmed_mcts={summary['ane_confirmed_mcts']}, "
        f"unsolved_blocked={summary['unsolved_blocked']}"
    )
    if summary["ane_score_margin_median"] is not None:
        print(
            f"ANE trace {label} margin: "
            f"min={summary['ane_score_margin_min']:.3f}, "
            f"median={summary['ane_score_margin_median']:.3f}, "
            f"max={summary['ane_score_margin_max']:.3f}"
        )
    reason_counts = summary["reason_counts"]
    if isinstance(reason_counts, dict) and reason_counts:
        reasons = ", ".join(
            f"{name}:{count}" for name, count in sorted(reason_counts.items())
        )
        print(f"ANE trace {label} reasons: {reasons}")
    examples = summary["blocked_examples"]
    if isinstance(examples, list) and examples:
        print(f"ANE trace {label} blocked examples:")
        for example in examples:
            print(f"  {example}")


def run(args: argparse.Namespace) -> int:
    baseline = load_results(args.baseline)
    candidate = load_results(args.candidate)
    pairs = pair_results(baseline, candidate, match_repeat_ids=args.match_repeat_ids)
    if len(pairs) < args.min_common:
        print(
            f"ERROR: only {len(pairs)} common puzzle ids, "
            f"below --min-common {args.min_common}",
            file=sys.stderr,
        )
        return 1

    base_common = [base for _, base, _ in pairs]
    cand_common = [cand for _, _, cand in pairs]
    base_solved = solved_count(base_common)
    cand_solved = solved_count(cand_common)
    base_acc = base_solved / len(pairs)
    cand_acc = cand_solved / len(pairs)
    solved_drop = base_solved - cand_solved
    accuracy_drop = base_acc - cand_acc

    print(
        f"Puzzle compare: baseline {base_solved}/{len(pairs)} "
        f"({base_acc:.2%}), candidate {cand_solved}/{len(pairs)} "
        f"({cand_acc:.2%})"
    )
    print(
        f"Delta: solved_drop={solved_drop}, accuracy_drop={accuracy_drop:.2%}, "
        f"errors baseline={error_count(base_common)}, "
        f"candidate={error_count(cand_common)}"
    )
    if args.ane_summary:
        print_ane_trace_summary(cand_common, "candidate")

    failures: list[str] = []
    if cand_solved < args.min_candidate_solved:
        failures.append(
            f"candidate solved {cand_solved}, below {args.min_candidate_solved}"
        )
    if solved_drop > args.max_solved_drop:
        failures.append(
            f"candidate lost {solved_drop} solved puzzles, "
            f"above {args.max_solved_drop}"
        )
    if accuracy_drop > args.max_accuracy_drop:
        failures.append(
            f"candidate accuracy dropped {accuracy_drop:.2%}, "
            f"above {args.max_accuracy_drop:.2%}"
        )
    if error_count(cand_common) > args.max_candidate_errors:
        failures.append(
            f"candidate had {error_count(cand_common)} errors, "
            f"above {args.max_candidate_errors}"
        )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--candidate", type=pathlib.Path, required=True)
    parser.add_argument("--min-common", type=int, default=1)
    parser.add_argument("--min-candidate-solved", type=int, default=0)
    parser.add_argument("--max-solved-drop", type=int, default=10)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.03)
    parser.add_argument("--max-candidate-errors", type=int, default=0)
    parser.add_argument(
        "--ane-summary",
        action="store_true",
        help="Print candidate Hybrid/ANE trace agreement and AB-block summaries.",
    )
    parser.add_argument(
        "--match-repeat-ids",
        action="store_true",
        help="Match candidate ids like puzzle#r2 to baseline puzzle ids.",
    )
    args = parser.parse_args(argv)
    args.min_common = max(1, args.min_common)
    args.max_solved_drop = max(0, args.max_solved_drop)
    args.max_accuracy_drop = max(0.0, args.max_accuracy_drop)
    args.max_candidate_errors = max(0, args.max_candidate_errors)
    args.min_candidate_solved = max(0, args.min_candidate_solved)
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv or sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
