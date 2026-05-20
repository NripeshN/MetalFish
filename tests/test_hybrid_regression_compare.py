#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import hybrid_regression_compare as compare  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def summary(scores, nps=1000, errors=None):
    return compare.EngineSummary(
        scores=scores,
        completed=[24 for _ in scores],
        score_mean=sum(scores) / len(scores),
        score_min=min(scores),
        score_max=max(scores),
        nps_median=nps,
        nodes_median=10000,
        elapsed_median_ms=2000,
        errors=errors or [],
    )


def test_regression_thresholds_allow_noise() -> None:
    args = argparse.Namespace(
        max_bk_mean_drop=0.67,
        max_bk_min_drop=1,
        min_candidate_bk_score=23,
        max_perf_regression=0.25,
    )
    failures = compare.compare_summaries(
        summary([24, 24, 24], nps=1000),
        summary([24, 23, 24], nps=900),
        24,
        args,
    )
    expect("one noisy miss allowed", failures == [])


def test_regression_thresholds_catch_accuracy_drop() -> None:
    args = argparse.Namespace(
        max_bk_mean_drop=0.67,
        max_bk_min_drop=1,
        min_candidate_bk_score=23,
        max_perf_regression=0.25,
    )
    failures = compare.compare_summaries(
        summary([24, 24, 24], nps=1000),
        summary([22, 23, 23], nps=900),
        24,
        args,
    )
    expect("accuracy drop detected", any("BK mean" in item for item in failures))
    expect("absolute floor detected", any("absolute floor" in item for item in failures))


def test_regression_thresholds_catch_perf_drop() -> None:
    args = argparse.Namespace(
        max_bk_mean_drop=0.67,
        max_bk_min_drop=1,
        min_candidate_bk_score=0,
        max_perf_regression=0.25,
    )
    failures = compare.compare_summaries(
        summary([24, 24, 24], nps=1000),
        summary([24, 24, 24], nps=700),
        24,
        args,
    )
    expect("perf drop detected", any("median NPS" in item for item in failures))


def test_select_positions_by_label() -> None:
    selected = compare.select_bk_positions("BK.07,BK.09")
    expect("two selected", len(selected) == 2)
    expect("labels", [item[2] for item in selected] == ["BK.07", "BK.09"])


def test_parse_args_defaults() -> None:
    args = compare.parse_args(
        [
            "--baseline-engine",
            "baseline/build/metalfish",
            "--candidate-engine",
            "build/metalfish",
            "--weights",
            "networks/BT4-1024x15x32h-swa-6147500.pb",
        ]
    )
    expect("repeat", args.repeat == 3)
    expect("bk movetime", args.bk_movetime == 5000)
    expect("perf movetime", args.perf_movetime == 2000)


def main() -> int:
    test_regression_thresholds_allow_noise()
    test_regression_thresholds_catch_accuracy_drop()
    test_regression_thresholds_catch_perf_drop()
    test_select_positions_by_label()
    test_parse_args_defaults()
    print("Hybrid regression compare tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
