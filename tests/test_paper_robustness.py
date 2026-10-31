#!/usr/bin/env python3
"""Pins tools/paper/robustness.py, the final-reading analyses of protocol amendment 12 (no games
run)."""

from __future__ import annotations

import datetime as dt
import math
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools" / "paper"))

try:
    import robustness  # noqa: E402
except ImportError as exc:  # python-chess missing
    robustness = None
    IMPORT_ERROR = exc

UTC = dt.timezone.utc
T0 = dt.datetime(2026, 10, 20, 10, 0, tzinfo=UTC)


def game(rnd, fen, white, black, result, start_min, term="normal"):
    start = T0 + dt.timedelta(minutes=start_min)
    end = start + dt.timedelta(minutes=2)
    fmt = "%Y-%m-%dT%H:%M:%S %z"
    return {
        "round": str(rnd),
        "fen": fen,
        "white": white,
        "black": black,
        "result": result,
        "termination": term,
        "comment_tail": "",
        "plycount": "80",
        "start_time": start.strftime(fmt),
        "end_time": end.strftime(fmt),
    }


def pair(rnd, fen, a_score_white, a_score_black, start_min, term=("normal", "normal")):
    res = {1.0: "1-0", 0.5: "1/2-1/2", 0.0: "0-1"}
    res_b = {1.0: "0-1", 0.5: "1/2-1/2", 0.0: "1-0"}
    return [
        game(rnd, fen, "A", "B", res[a_score_white], start_min, term[0]),
        game(rnd, fen, "B", "A", res_b[a_score_black], start_min + 3, term[1]),
    ]


@unittest.skipIf(robustness is None, "robustness unavailable (python-chess missing)")
class RobustnessTest(unittest.TestCase):
    def test_elo_of_scores(self):
        self.assertEqual(robustness.elo_of_scores([0.5] * 8)["elo"], 0.0)
        st = robustness.elo_of_scores([1.0, 0.5])
        self.assertAlmostEqual(st["elo"], 400 * math.log10(3))
        self.assertIsNone(robustness.elo_of_scores([])["elo"])

    def test_pair_records_follow_play_order(self):
        games = pair(1, "f1", 1.0, 0.5, 0) + pair(2, "f2", 0.0, 0.5, 10)
        recs = robustness.pair_records(games, "A")
        self.assertEqual([r["fen"] for r in recs], ["f1", "f2"])
        self.assertEqual([r["score"] for r in recs], [0.75, 0.25])

    def test_load_regression_intercept_is_the_zero_load_estimate(self):
        games, samples = [], []
        # ten unexposed pairs scoring 0.5 and ten fully exposed pairs scoring 0.25
        for k in range(20):
            exposed = k >= 10
            games += pair(
                k + 1, f"f{k}", 0.5 if exposed else 1.0, 0.0 if exposed else 0.0, 10 * k
            )
            for m in range(10 * k, 10 * k + 6):
                samples.append(
                    (T0 + dt.timedelta(minutes=m, seconds=30), 500 if exposed else 50)
                )
        pairs = robustness.pair_records(games, "A")
        out = robustness.load_regression(games, pairs, samples, threshold=300)
        self.assertEqual(out["pairs_with_samples"], 20)
        self.assertEqual(out["pairs_exposed"], 10)
        self.assertAlmostEqual(out["fit"]["a"], 0.5)
        self.assertAlmostEqual(out["fit"]["b"], -0.25)
        self.assertAlmostEqual(out["elo_at_zero_load"], 0.0)

    def test_forfeit_variants(self):
        games = pair(1, "f1", 1.0, 0.5, 0) + pair(
            2, "f2", 0.0, 0.5, 10, term=("time forfeit", "normal")
        )
        v = robustness.forfeit_variants(games, "A")
        self.assertEqual(v["forfeit_games"], 1)
        self.assertEqual(v["removed"]["pairs"], 1)
        self.assertAlmostEqual(v["removed"]["score"], 0.75)
        self.assertAlmostEqual(v["as_draws"]["score"], (0.75 + 0.5) / 2)

    def test_without_first_openings(self):
        games = (
            pair(1, "f1", 1, 1, 0) + pair(2, "f2", 1, 1, 10) + pair(3, "f3", 0, 0, 20)
        )
        rest = robustness.without_first_openings(games, 2)
        self.assertEqual({g["fen"] for g in rest}, {"f3"})

    def test_chi2_and_cochran_q(self):
        self.assertAlmostEqual(robustness.chi2_sf(7.814727903, 3), 0.05, places=6)
        self.assertAlmostEqual(robustness.chi2_sf(2.0, 1), 0.157299207, places=6)
        self.assertAlmostEqual(robustness.chi2_sf(0.5, 4), 0.973500981, places=6)
        q = robustness.cochran_q([0.0, 10.0], [5.0, 5.0])
        self.assertAlmostEqual(q["q"], 2.0)
        self.assertEqual(q["df"], 1)
        self.assertAlmostEqual(q["pooled"], 5.0)
        same = robustness.cochran_q([3.0, 3.0, 3.0, 3.0], [4.0, 5.0, 6.0, 7.0])
        self.assertAlmostEqual(same["q"], 0.0)
        self.assertAlmostEqual(same["p"], 1.0)

    def test_chunks_from_history_and_from_round_restarts(self):
        games = []
        for k in range(6):
            games += pair(k % 3 + 1, f"f{k}", 1.0, 0.0, 10 * k)
        pairs = robustness.pair_records(games, "A")
        self.assertEqual(robustness.chunk_of_pairs(pairs, [0, 6]), [0, 0, 0, 1, 1, 1])
        self.assertEqual(robustness.chunk_of_pairs(pairs, []), [0, 0, 0, 1, 1, 1])

    def test_chunk_bootstrap_resamples_whole_chunks(self):
        # every chunk has the same mean, so resampling chunks cannot move the estimate
        pairs = [{"score": s} for s in (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)]
        out = robustness.chunk_bootstrap(pairs, [0, 0, 1, 1, 2, 2], n_boot=200)
        self.assertEqual(out["chunks"], 3)
        self.assertAlmostEqual(out["ci95"][0], 0.0)
        self.assertAlmostEqual(out["ci95"][1], 0.0)

    def test_linear_combination_includes_shared_opening_covariance(self):
        scores = [0.25, 0.75, 0.5, 1.0, 0.25, 0.5, 0.75, 0.5]
        cell = [{"fen": f"f{k}", "score": s} for k, s in enumerate(scores)]
        one = robustness.elo_of_scores(scores)
        both = robustness.linear_combination([(cell, 1.0), (cell, 1.0)])
        self.assertAlmostEqual(both["elo"], 2 * one["elo"])
        s = one["score"]
        d = 400.0 / (math.log(10.0) * s * (1 - s))
        # identical pair scores on all shared openings: the SE doubles instead of growing by sqrt(2)
        self.assertAlmostEqual(both["elo_pm"], 1.96 * 2 * d * one["se"])
        self.assertEqual(both["shared_openings"], [len(scores)])
        other = [{"fen": f"g{k}", "score": v} for k, v in enumerate(scores)]
        indep = robustness.linear_combination([(cell, 1.0), (other, 1.0)])
        self.assertAlmostEqual(indep["elo_pm"], 1.96 * math.sqrt(2) * d * one["se"])

    def test_load_coverage_per_block(self):
        games = pair(1, "f1", 1, 1, 0) + pair(2, "f2", 1, 1, 100)
        samples = [(T0 + dt.timedelta(minutes=1), 10)]
        pairs = robustness.pair_records(games, "A")
        cov = robustness.load_coverage(games, pairs, samples, [0, 1])
        self.assertEqual(cov, {0: (1, 1), 1: (0, 1)})

    def test_move_comments_stats_and_adjudications(self):
        self.assertEqual(
            robustness.parse_move_comment("+1.06/25 1.180s, n=8681914, nps=7357554"),
            {"depth": 25, "time": 1.18, "nodes": 8681914, "nps": 7357554},
        )
        self.assertEqual(
            robustness.parse_move_comment(
                "-6.59/26 0.296s, n=1, nps=2, Black wins by adjudication"
            )["depth"],
            26,
        )
        self.assertIsNone(robustness.parse_move_comment("book"))
        pgn = (
            '[Event "t"]\n[White "A"]\n[Black "B"]\n[Result "1-0"]\n\n'
            "1. e4 {+0.30/20 1.000s, n=100, nps=100} e5 {-0.20/18 0.500s, n=50, nps=100} "
            "2. Nf3 {+0.31/21 3.000s, n=300, nps=100} 1-0\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".pgn", delete=False) as fh:
            fh.write(pgn)
            path = fh.name
        moves = list(robustness.iter_engine_moves(path, with_fen=True))
        self.assertEqual(
            [(m["engine"], m["index"]) for m in moves], [("A", 1), ("B", 1), ("A", 2)]
        )
        self.assertTrue(moves[1]["fen"].startswith("rnbqkbnr/pppppppp/8/8/4P3/"))
        st = robustness.move_stats(
            moves, classify=lambda m: "first" if m["index"] == 1 else "later"
        )
        self.assertEqual(st[("A", "all")]["moves"], 2)
        self.assertAlmostEqual(st[("A", "all")]["time_mean"], 2.0)
        self.assertEqual(st[("A", "first")]["moves"], 1)
        self.assertEqual(st[("A", "all")]["time_by_index"], {"1-10": 2.0})
        games = [
            game(1, "f", "A", "B", "1/2-1/2", 0, "adjudication"),
            game(1, "f", "B", "A", "0-1", 3, "adjudication"),
            game(2, "g", "A", "B", "1-0", 6),
        ]
        self.assertEqual(
            robustness.adjudications(games, ["A", "B"]),
            {"draws": 1, "wins": {"A": 1, "B": 0}},
        )

    def test_flag_counts_on_disagreements(self):
        recs = [
            {
                "reason": "engines_agree",
                "ab_move": "e2e4",
                "mcts_move": "e2e4",
                "x_flag": 1,
            },
            {"reason": "ab_fallback_time_safety", "x_flag": 1},
            {
                "reason": "ab_root_rejects_mcts",
                "ab_move": "e2e4",
                "mcts_move": "d2d4",
                "x_flag": 1,
                "visits": 1,
                "ab_hints": [{"move": "d2d4"}],
            },
            {
                "reason": "ab_default",
                "ab_move": "e2e4",
                "mcts_move": "d2d4",
                "x_flag": 0,
                "visits": 37,
                "ab_hints": [],
            },
        ]
        out = robustness.flag_counts(recs)
        self.assertEqual(out["disagreements"], 2)
        self.assertEqual(out["flags"], {"x_flag": 1})
        self.assertEqual((out["with_hints"], out["first_hint_is_mcts_move"]), (1, 1))


@unittest.skipIf(robustness is None, "robustness unavailable (python-chess missing)")
class BuildSplitTest(unittest.TestCase):
    def test_pairs_follow_the_build_of_their_first_game(self):
        import json

        builds_json = {
            "builds": [
                {"build": "beta", "installed_utc": "2026-10-01T00:00:00Z"},
                {"build": "rc", "installed_utc": "2026-10-20T10:04:00Z"},
            ]
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(builds_json, fh)
            path = fh.name
        builds = robustness.load_builds(path)
        self.assertEqual([b for _, b in builds], ["beta", "rc"])
        # T0 is 2026-10-20 10:00 UTC: the first pair starts on the beta and finishes on the rc,
        # the second pair starts on the rc; a lone game at T0 + 20 min is unpaired
        games = pair(1, "f1", 1.0, 0.5, 0) + pair(2, "f2", 0.0, 0.5, 10)
        games.append(game(3, "f3", "A", "B", "1-0", 20))
        split = robustness.split_by_build(games, builds)
        self.assertEqual([g["fen"] for g in split["beta"]], ["f1", "f1"])
        self.assertEqual([g["fen"] for g in split["rc"]], ["f2", "f2", "f3"])
        import penta_stats

        st = penta_stats.compute(split["rc"], "A")
        self.assertEqual((st["pairs"], st["unpaired_games"]), (1, 1))


class BackendbenchParseTest(unittest.TestCase):
    def test_parses_the_row_of_the_requested_batch(self):
        import microbench

        txt = (
            "size, mean nps, mean ms,   sdev,     cv, max nps,  median, min nps,\n"
            "   8,      410,   19.51, 0.4000, 0.0205,     430,     409,     380\n"
        )
        row = microbench.parse_backendbench(txt, 8)
        self.assertEqual(
            row, {"batch": 8, "mean_eps": 410.0, "mean_ms": 19.51, "median_eps": 409.0}
        )
        self.assertIn("error", microbench.parse_backendbench("weights not found", 8))


if __name__ == "__main__":
    unittest.main()
