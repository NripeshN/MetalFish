#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import sprt_test  # noqa: E402
from tools import tuning_workflow as tw  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def test_load_candidate_config_parses_and_defaults() -> None:
    config = {
        "baseline_options": {"MultiPV": "1"},
        "elo0": 0.0,
        "elo1": 4.0,
        "candidates": [
            {"label": "a", "options": {"MCTSContempt": "100"}},
            {
                "label": "b",
                "options": {"MCTSContempt": "300"},
                "elo1": 8.0,
                "note": "hi",
            },
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(config, handle)
        path = handle.name
    baseline, candidates = tw.load_candidate_config(path)
    expect("baseline overrides parsed", baseline == {"MultiPV": "1"})
    expect("two candidates", len(candidates) == 2)
    expect("default elo1 applied", candidates[0].elo1 == 4.0)
    expect("per-candidate elo1 overrides default", candidates[1].elo1 == 8.0)
    expect("note captured", candidates[1].note == "hi")
    expect("options stringified", candidates[0].options == {"MCTSContempt": "100"})


def test_load_candidate_config_rejects_bad_input() -> None:
    bad_configs = [
        {"candidates": []},
        {"candidates": [{"label": "x"}]},  # no options
        {
            "candidates": [
                {"label": "dup", "options": {"A": "1"}},
                {"label": "dup", "options": {"A": "2"}},
            ]
        },
    ]
    for cfg in bad_configs:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(cfg, handle)
            path = handle.name
        raised = False
        try:
            tw.load_candidate_config(path)
        except ValueError:
            raised = True
        expect(f"rejects invalid config {cfg}", raised)


def test_base_options_for_mode() -> None:
    ab = tw.base_options_for_mode("ab", "w.pb", 4, 1024)
    expect("ab disables hybrid", ab["UseHybridSearch"] == "false")
    expect("ab disables mcts", ab["UseMCTS"] == "false")
    expect("hash applied", ab["Hash"] == "1024")

    mcts = tw.base_options_for_mode("mcts", "w.pb", 4, 512)
    expect("mcts enables mcts", mcts["UseMCTS"] == "true")
    expect("mcts weights set", mcts["NNWeights"] == "w.pb")

    hybrid = tw.base_options_for_mode("hybrid", "w.pb", 4, 2048)
    expect("hybrid enables hybrid", hybrid["UseHybridSearch"] == "true")
    expect("hybrid hash applied", hybrid["Hash"] == "2048")

    raised = False
    try:
        tw.base_options_for_mode("nope", "w.pb", 4, 16)
    except ValueError:
        raised = True
    expect("unknown mode rejected", raised)


def _outcome(
    label: str, elo: float, lo: float, status: str, games: int = 100
) -> tw.CandidateOutcome:
    wins = games // 3
    draws = games // 3
    losses = games - wins - draws
    return tw.CandidateOutcome(
        label=label,
        options={"MCTSContempt": label},
        wins=wins,
        draws=draws,
        losses=losses,
        elo=elo,
        elo_lo=lo,
        elo_hi=elo + (elo - lo),
        status=status,
        games=games,
    )


def test_rank_outcomes_prefers_pass_then_elo() -> None:
    outcomes = [
        _outcome("neutral", 1.0, -3.0, "max_games"),
        _outcome("passed", 4.0, 1.0, "H1"),
        _outcome("failed", 9.0, 5.0, "H0"),
        _outcome("best_elo", 7.0, -1.0, "max_games"),
    ]
    ranked = tw.rank_outcomes(outcomes)
    expect("SPRT pass ranks first", ranked[0].label == "passed")
    expect("SPRT fail ranks last", ranked[-1].label == "failed")
    middle = [o.label for o in ranked[1:3]]
    expect(
        "higher Elo ranks above lower among undecided",
        middle == ["best_elo", "neutral"],
    )


def test_recommend_requires_statistical_gain() -> None:
    # No candidate has a positive lower bound / pass -> keep baseline.
    weak = [
        _outcome("c1", 3.0, -2.0, "max_games"),
        _outcome("c2", 1.0, -5.0, "max_games"),
    ]
    expect("no recommendation when unproven", tw.recommend(weak) is None)

    strong = [
        _outcome("c1", 3.0, -2.0, "max_games"),
        _outcome("c2", 6.0, 1.5, "H1"),
    ]
    rec = tw.recommend(strong)
    expect("recommends validated candidate", rec is not None and rec.label == "c2")

    # Positive lower bound but below the min-games bar -> not recommended.
    tiny = [_outcome("c1", 8.0, 2.0, "max_games", games=10)]
    expect("min games gate enforced", tw.recommend(tiny, min_games=40) is None)


def test_outcome_from_sprt_and_report_smoke() -> None:
    result = sprt_test.SPRTResult(wins=30, draws=40, losses=20)
    result.elo_est = 12.0
    result.elo_ci_lo = 2.0
    result.elo_ci_hi = 22.0
    result.status = "H1"
    result.llr = 3.1
    candidate = tw.Candidate(label="contempt-300", options={"MCTSContempt": "300"})
    outcome = tw.outcome_from_sprt(candidate, result)
    expect("wins mapped", outcome.wins == 30)
    expect("draw ratio computed", abs(outcome.draw_ratio - 40 / 90) < 1e-9)
    expect("score pct computed", abs(outcome.score_pct - 100 * 50 / 90) < 1e-6)
    report = tw.format_report([outcome], outcome)
    expect("report mentions candidate", "contempt-300" in report)
    expect("report mentions recommendation", "RECOMMENDED" in report)


def main() -> int:
    test_load_candidate_config_parses_and_defaults()
    test_load_candidate_config_rejects_bad_input()
    test_base_options_for_mode()
    test_rank_outcomes_prefers_pass_then_elo()
    test_recommend_requires_statistical_gain()
    test_outcome_from_sprt_and_report_smoke()
    print("test_tuning_workflow: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
