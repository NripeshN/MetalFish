#!/usr/bin/env python3
"""Pins tools/paper/penta_stats.py and tools/paper/parse_hybrid_trace.py (no games run)."""

from __future__ import annotations

import json
import math
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "paper"))

import parse_hybrid_trace  # noqa: E402
import penta_stats  # noqa: E402


def pgn_game(
    rnd: int, white: str, black: str, result: str, term: str = "normal"
) -> str:
    return (
        f'[Event "t"]\n[Round "{rnd}"]\n[White "{white}"]\n[Black "{black}"]\n[Result "{result}"]\n'
        f'[Termination "{term}"]\n[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]\n[SetUp "1"]\n\n1. e4 {result}\n\n'
    )


class PentaStatsTest(unittest.TestCase):
    def _stats(self, text: str, engine: str = "A"):
        with tempfile.NamedTemporaryFile("w", suffix=".pgn", delete=False) as fh:
            fh.write(text)
            path = fh.name
        return penta_stats.compute(penta_stats.read_games(path), engine)

    def test_reproduces_fastchess_symmetric_case(self):
        # Ptnml [0, 10, 20, 10, 0]: fastchess printed Elo -0.00 +/- 38.22, nElo 0.00 +/- 76.13
        games = []
        rnd = 1
        for _ in range(10):  # LD pairs: A loses one, draws one
            games.append(pgn_game(rnd, "A", "B", "0-1"))
            games.append(pgn_game(rnd, "B", "A", "1/2-1/2"))
            rnd += 1
        for _ in range(20):  # DD pairs
            games.append(pgn_game(rnd, "A", "B", "1/2-1/2"))
            games.append(pgn_game(rnd, "B", "A", "1/2-1/2"))
            rnd += 1
        for _ in range(10):  # WD pairs
            games.append(pgn_game(rnd, "A", "B", "1-0"))
            games.append(pgn_game(rnd, "B", "A", "1/2-1/2"))
            rnd += 1
        s = self._stats("".join(games))
        self.assertEqual(s["ptnml_0_2"], [0, 10, 20, 10, 0])
        self.assertAlmostEqual(s["elo"], 0.0, places=6)
        self.assertAlmostEqual(s["elo_pm"], 38.22, delta=0.05)
        self.assertAlmostEqual(s["nelo_pm"], 76.13, delta=0.05)
        self.assertAlmostEqual(s["los"], 0.5, places=6)
        self.assertAlmostEqual(s["draw_ratio_pairs"], 0.5)
        self.assertEqual(s["pairs"], 40)

    def test_reproduces_fastchess_asymmetric_case(self):
        # Ptnml [2, 12, 19, 7, 0] with 18W/35D/27L: fastchess printed LOS 3.59 %, PairsRatio 0.50,
        # DrawRatio 47.50 % (pairs), WL/DD 1.38; penta_stats computed Elo -39.25 +/- 43.32
        games = []
        rnd = 1

        def add(res_a_white: str, res_a_black_as_black: str):
            nonlocal rnd
            games.append(pgn_game(rnd, "A", "B", res_a_white))
            games.append(pgn_game(rnd, "B", "A", res_a_black_as_black))
            rnd += 1

        for _ in range(2):
            add("0-1", "1-0")  # LL
        for _ in range(12):
            add("0-1", "1/2-1/2")  # LD
        for _ in range(11):
            add("1-0", "1-0")  # WL
        for _ in range(8):
            add("1/2-1/2", "1/2-1/2")  # DD
        for _ in range(7):
            add("1-0", "1/2-1/2")  # WD
        s = self._stats("".join(games))
        self.assertEqual(s["ptnml_0_2"], [2, 12, 19, 7, 0])
        self.assertEqual((s["wins"], s["draws"], s["losses"]), (18, 35, 27))
        self.assertAlmostEqual(100 * s["los"], 3.59, delta=0.05)
        self.assertAlmostEqual(s["pairs_ratio"], 0.5)
        self.assertAlmostEqual(s["wl_dd_ratio"], 11 / 8, places=6)
        self.assertAlmostEqual(s["elo"], -39.25, delta=0.05)
        self.assertAlmostEqual(s["elo_pm"], 43.32, delta=0.05)

    def test_unpaired_and_terminations(self):
        text = pgn_game(1, "A", "B", "1-0", "time forfeit") + pgn_game(
            2, "A", "B", "*", "unterminated"
        )
        s = self._stats(text)
        self.assertEqual(s["pairs"], 0)
        self.assertEqual(s["terminations"].get("time_forfeit"), 1)
        self.assertEqual(s["unfinished"], 1)


class TraceParserTest(unittest.TestCase):
    SPLIT = "info string Hybrid thread split: MCTS=1 AB=6 (search workers=7, +1 coordinator, total=8)"
    TRACE = (
        "info string HybridTrace: reason=ab_root_rejects_mcts mode=dynamic selected=g3e4 ABMove=g3e4 "
        "MCTSMove=h2h4 ABScore=43 ABDepth=10 MCTSQ=-0.175 MCTSCP=-52 MCTSPlayouts=37 MCTSEvals=37 "
        "MCTSBestVisits=24 MCTSRootVisits=36 MCTSTop=[b1d2:n=24:cn=24:q=0.153:p=0.575,f3d2:n=8:cn=8:q=0.093:p=0.234] "
        "ABRoot=[g3e4:s=172:ps=172:avg=169:lb=0:ub=0:eff=430369] ANETop=none"
    )
    FALLBACK = "info string Time safety: estimated move budget 388ms; using Alpha-Beta without transformer MCTS"

    def _parse(self, lines):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write("\n".join(lines) + "\n")
            path = fh.name
        return parse_hybrid_trace.parse(path)

    def test_parses_trace_fields_and_lists(self):
        recs = self._parse([self.SPLIT, self.TRACE])
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(r["reason"], "ab_root_rejects_mcts")
        self.assertEqual(r["ab_move"], "g3e4")
        self.assertEqual(r["mcts_move"], "h2h4")
        self.assertEqual(r["mcts_evals"], 37)
        self.assertEqual(r["ab_depth"], 10)
        self.assertAlmostEqual(r["mcts_q"], -0.175)
        self.assertTrue(r["gpu_active"])
        self.assertEqual(r["thread_split"]["ab"], 6)
        self.assertEqual(r["mcts_top"][0]["move"], "b1d2")
        self.assertEqual(r["mcts_top"][0]["n"], 24)
        self.assertEqual(r["ab_root"][0]["eff"], 430369)
        self.assertIsNone(r["ane_top"])

    def test_fallback_and_summary(self):
        recs = self._parse([self.SPLIT, self.FALLBACK, self.TRACE, self.FALLBACK])
        self.assertEqual(len(recs), 3)
        self.assertEqual(recs[0]["reason"], "ab_fallback_time_safety")
        self.assertEqual(recs[0]["fallback_budget_ms"], 388)
        summ = parse_hybrid_trace.summarise(recs)
        self.assertEqual(summ["moves_total"], 3)
        self.assertEqual(summ["moves_fallback"], 2)
        self.assertAlmostEqual(summ["gpu_active_fraction"], 1 / 3)
        self.assertEqual(summ["disagreements"], 1)
        self.assertEqual(summ["mcts_move_selected_on_disagreement"], 0)

    def test_fastchess_log_line_format(self):
        line = (
            "[Engine] [13:59:29.704896] <         0x16fdff000>  MF-Hybrid-T8 ---> "
            + self.TRACE
        )
        recs = self._parse([line])
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["engine"], "MF-Hybrid-T8")

    def test_rounds_restarting_per_invocation_pair_by_round_and_fen(self):
        # Two fastchess invocations (block 0 and a resume/block 1) both number their rounds from 1;
        # the opening FEN keeps their pairs apart. An orphan game (one game of a round) is unpaired.
        f1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        f2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        f3 = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"

        def game(rnd, fen, white, black, result):
            return (
                f'[Event "t"]\n[Round "{rnd}"]\n[White "{white}"]\n[Black "{black}"]\n[Result "{result}"]\n'
                f'[Termination "normal"]\n[FEN "{fen}"]\n[SetUp "1"]\n\n1. e4 {result}\n\n'
            )

        text = (
            game(1, f1, "A", "B", "1-0")
            + game(1, f1, "B", "A", "1/2-1/2")  # block 0, round 1: A scores 1.5
            + game(2, f2, "A", "B", "1-0")  # orphan: second game never played
            + game(1, f3, "A", "B", "0-1")
            + game(
                1, f3, "B", "A", "1/2-1/2"
            )  # next invocation, round 1 again: A scores 0.5
        )
        st = self._stats(text, "A")
        self.assertEqual(st["games"], 5)
        self.assertEqual(st["pairs"], 2)
        self.assertEqual(st["unpaired_games"], 1)
        self.assertEqual(st["ptnml_0_2"], [0, 1, 0, 1, 0])


class OracleSamplingTest(unittest.TestCase):
    """Pins the stratified sampler of tools/paper/oracle_probe.py (protocol §11.6)."""

    OPENING = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 10"
    MIDDLE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 30"
    ENDGAME = "8/8/4k3/8/8/4K3/4R3/8 w - - 0 60"

    def setUp(self):
        try:
            import oracle_probe  # noqa: F401
        except ImportError as exc:  # python-chess missing
            self.skipTest(f"oracle_probe unavailable: {exc}")
        self.op = sys.modules["oracle_probe"]

    def test_phase_rule(self):
        self.assertEqual(self.op.phase_of(self.OPENING), "opening")
        self.assertEqual(self.op.phase_of(self.MIDDLE), "middlegame")
        self.assertEqual(self.op.phase_of(self.ENDGAME), "endgame")
        self.assertEqual(self.op.phase_of(self.ENDGAME, endgame_pieces=0), "middlegame")

    def _records(self):
        recs = []
        for i in range(300):
            fen = (self.OPENING, self.MIDDLE, self.ENDGAME)[i % 3]
            recs.append(
                {
                    "fen": fen,
                    "ab_move": "e2e4",
                    "mcts_move": "d2d4",
                    "mcts_root_visits": i + 1,
                }
            )
        return recs

    def test_stratified_sample_is_proportional_and_deterministic(self):
        sample, info = self.op.stratified_sample(self._records(), 60, seed=1)
        self.assertEqual(len(sample), 60)
        self.assertEqual(info["method"], "stratified")
        self.assertEqual(sum(info["strata_sample"].values()), 60)
        self.assertEqual(sum(info["strata_population"].values()), 300)
        by_phase = {}
        for r in sample:
            self.assertIn("_stratum", r)
            phase = r["_stratum"].split("/")[0]
            by_phase[phase] = by_phase.get(phase, 0) + 1
        for phase in ("opening", "middlegame", "endgame"):
            self.assertAlmostEqual(by_phase[phase], 20, delta=1)
        bins = {r["_stratum"].split("/")[1] for r in sample}
        self.assertEqual(bins, {"visits_low", "visits_mid", "visits_high"})
        again, _ = self.op.stratified_sample(self._records(), 60, seed=1)
        self.assertEqual(
            [r["mcts_root_visits"] for r in sample],
            [r["mcts_root_visits"] for r in again],
        )
        other, _ = self.op.stratified_sample(self._records(), 60, seed=2)
        self.assertNotEqual(
            [r["mcts_root_visits"] for r in sample],
            [r["mcts_root_visits"] for r in other],
        )

    def test_sample_capped_at_population(self):
        sample, info = self.op.stratified_sample(self._records()[:10], 600, seed=1)
        self.assertEqual(len(sample), 10)
        self.assertEqual(info["requested"], 10)


if __name__ == "__main__":
    unittest.main()
