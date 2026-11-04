#!/usr/bin/env python3
"""Pins tools/paper/penta_stats.py and tools/paper/parse_hybrid_trace.py (no games run)."""

from __future__ import annotations

import contextlib
import io
import json
import math
import pathlib
import re
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

    def test_time_forfeits_by_arm(self):
        # the side that lost on time: Black after 1-0, White after 0-1 (protocol amendment 10)
        text = (
            pgn_game(1, "A", "B", "1-0", "time forfeit")
            + pgn_game(1, "B", "A", "1-0", "time forfeit")
            + pgn_game(2, "A", "B", "0-1", "time forfeit")
            + pgn_game(2, "B", "A", "1/2-1/2")
        )
        s = self._stats(text)
        self.assertEqual(s["time_forfeits_by"], {"B": 1, "A": 2})
        self.assertEqual(s["terminations"]["time_forfeit"], 3)


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

    def test_game_ids_from_position_commands(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        head = "[Engine] [10:00:00.000000] <         0x16fdff000>  "

        def to(name, payload):
            return f"{head}{name} <--- {payload}"

        def frm(name, payload):
            return f"{head}{name} ---> {payload}"

        recs = self._parse(
            [
                to("MF", f"position fen {fen}"),
                frm("MF", self.TRACE),
                to("AB", f"position fen {fen} moves e7e5"),  # the opponent's own count
                to("MF", f"position fen {fen} moves e7e5 g1f3"),
                frm("MF", self.TRACE),
                to(
                    "MF", f"position fen {fen} moves d7d5"
                ),  # colour-reversed partner game
                frm("MF", self.FALLBACK),
                to("MF", "position startpos"),
                frm("MF", self.TRACE),
            ]
        )
        self.assertEqual([r["game"] for r in recs], [1, 1, 2, 3])
        self.assertEqual([r["opening"] for r in recs], [fen, fen, fen, "startpos"])
        self.assertEqual(recs[2]["reason"], "ab_fallback_time_safety")
        self.assertTrue(recs[1]["fen"].startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/"))
        # the CLI streams the same records to --out
        with tempfile.TemporaryDirectory() as d:
            log = pathlib.Path(d) / "engines.log"
            out = pathlib.Path(d) / "moves.jsonl"
            log.write_text(
                "\n".join(
                    [
                        to("MF", f"position fen {fen}"),
                        frm("MF", self.TRACE),
                        to("MF", "position startpos"),
                        frm("MF", self.TRACE),
                    ]
                )
                + "\n"
            )
            self.assertEqual(parse_hybrid_trace.main([str(log), "--out", str(out)]), 0)
            streamed = [json.loads(l) for l in out.read_text().splitlines()]
            self.assertEqual(streamed, parse_hybrid_trace.parse(str(log)))
            self.assertEqual([r["game"] for r in streamed], [1, 2])

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
        with tempfile.NamedTemporaryFile("w", suffix=".pgn", delete=False) as fh:
            fh.write(text)
            path = fh.name
        st = penta_stats.compute(penta_stats.read_games(path), "A")
        self.assertEqual(st["games"], 5)
        self.assertEqual(st["pairs"], 2)
        self.assertEqual(st["unpaired_games"], 1)
        self.assertEqual(st["ptnml_0_2"], [0, 1, 0, 1, 0])


class RunnerBookkeepingTest(unittest.TestCase):
    """Pins the runner's invocation history and resume arithmetic and the figure generator's
    use of them (blocks and resumes of one cell; an unpaired game left by an interruption).
    """

    def setUp(self):
        import make_figures
        import run_cell

        self.rc, self.mf = run_cell, make_figures
        self._saved = (run_cell.uci_id, run_cell.sha256, run_cell.run_capture)
        run_cell.uci_id = lambda *a, **k: "id name Test"
        run_cell.sha256 = lambda p: "ab" * 32
        run_cell.run_capture = lambda *a, **k: ""

    def tearDown(self):
        self.rc.uci_id, self.rc.sha256, self.rc.run_capture = self._saved

    def test_games_done_in_block(self):
        f = self.rc.games_done_in_block
        self.assertEqual(
            f(283, [], 0, 250), 283
        )  # no history: earlier blocks assumed complete
        hist = [{"block": 0, "games_before": 0}, {"block": 0, "games_before": 283}]
        self.assertEqual(
            f(501, hist, 1, 250), 0
        )  # new block after an orphan game: fresh
        self.assertEqual(f(283, hist, 0, 250), 283)
        self.assertEqual(
            f(601, hist + [{"block": 1, "games_before": 501}], 1, 250), 100
        )

    def test_manifest_keeps_history_and_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d)
            (out / "book.epd").write_text("x\n")
            cell = {
                "id": "T",
                "a": {"engine": "E", "threads": 8},
                "b": {"engine": "E", "threads": 6},
            }
            defaults = {"book": str(out / "book.epd"), "manager": "fastchess"}
            engines = {"E": {"cmd": "/bin/echo", "options": {}}}
            w = self.rc.write_manifest
            w(
                cell,
                defaults,
                engines,
                out,
                ["fc", "a"],
                5,
                0,
                "",
                games_before=0,
                start_opening=1,
            )
            self.rc.note_resume(
                out / "manifest.json", 0, {"games_in_pgn_before_resume": 0}
            )
            w(
                cell,
                defaults,
                engines,
                out,
                ["fc", "b"],
                3,
                0,
                "",
                games_before=4,
                start_opening=3,
            )
            w(
                cell,
                defaults,
                engines,
                out,
                ["fc", "c"],
                5,
                1,
                "",
                games_before=10,
                start_opening=6,
            )
            m = json.load(open(out / "manifest.json"))
            self.assertEqual([h["games_before"] for h in m["history"]], [0, 4, 10])
            self.assertEqual(len(m["resumes"]), 1)
            self.assertEqual(m["block"], 1)
            (out / "manifest.json").write_text("{broken")
            w(
                cell,
                defaults,
                engines,
                out,
                ["fc"],
                1,
                2,
                "",
                games_before=11,
                start_opening=11,
            )
            self.assertEqual(
                [h["block"] for h in json.load(open(out / "manifest.json"))["history"]],
                [2],
            )

    def test_cell_blocks_set_the_start_openings(self):
        # a cell's own "blocks" overrides the default, so a cell split into one block per
        # runner (amendment 15) plays contiguous slices of the same seeded opening order
        spec = {
            "defaults": {
                "manager": "fc",
                "book": "b.epd",
                "book_format": "epd",
                "book_order": "random",
                "seed": 1,
                "concurrency": 1,
                "games_per_round": 2,
                "draw": "movenumber=34",
                "resign": "movecount=3",
                "maxmoves": 200,
                "hash_mb": 2048,
                "report": "penta=true",
                "ratinginterval": 50,
                "blocks": 4,
            },
            "engines": {"E": {"cmd": "/bin/echo", "options": {}}},
            "cells": [
                {
                    "id": "T-R",
                    "tc": "30+0.3",
                    "pairs": 500,
                    "blocks": 10,
                    "hash_mb": 1024,
                    "a": {"engine": "E", "threads": 3},
                    "b": {"engine": "E", "threads": 1},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as d:
            m = pathlib.Path(d) / "exp.json"
            m.write_text(json.dumps(spec))
            saved = self.rc.RESULTS_ROOT
            self.rc.RESULTS_ROOT = pathlib.Path(d)
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    rc = self.rc.main(["T-R", "--manifest", str(m), "--dry-run"])
            finally:
                self.rc.RESULTS_ROOT = saved
        self.assertEqual(rc, 0)
        blocks = re.findall(
            r"block (\d+): (\d+) pairs, openings start=(\d+)", buf.getvalue()
        )
        self.assertEqual(len(blocks), 10)
        self.assertEqual(blocks[0], ("0", "50", "1"))
        self.assertEqual(blocks[3], ("3", "50", "151"))
        self.assertEqual(blocks[9], ("9", "50", "451"))
        self.assertIn("option.Hash=1024", buf.getvalue())

    def test_block_ranges_and_game_time(self):
        hist = {
            "history": [
                {"block": 0, "games_before": 0},
                {"block": 0, "games_before": 283},
                {"block": 1, "games_before": 501},
            ]
        }
        self.assertEqual(
            self.mf.block_ranges(hist, 700, 250, 4), [(0, 0, 501), (1, 501, 700)]
        )
        self.assertEqual(
            self.mf.block_ranges({}, 600, 250, 4), [(0, 0, 500), (1, 500, 600)]
        )
        with tempfile.NamedTemporaryFile("w", suffix=".pgn", delete=False) as fh:
            fh.write('[GameDuration "00:01:30"]\n[GameDuration "01:00:00"]\n')
            path = fh.name
        self.assertEqual(self.mf.pgn_game_seconds(pathlib.Path(path)), 3690.0)
        self.assertEqual(self.mf.pgn_game_durations(pathlib.Path(path)), (3690.0, 2))

    def test_process_labels_of_cut_paths(self):
        label = self.mf.process_label
        self.assertEqual(
            label(
                "/opt/homebrew/Cellar/python@3.11/3.11.17/Frameworks/Python.framework/Versio"
            ),
            "Python 3.11 (Homebrew)",
        )
        self.assertEqual(
            label(
                "/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/Me"
            ),
            "Spotlight indexing (Metadata.framework)",
        )
        self.assertEqual(
            label(
                "/System/Library/Frameworks/Virtualization.framework/Versions/A/XPCServices"
            ),
            "Virtualization.framework (virtual machine)",
        )
        self.assertEqual(
            label("/Applications/Kiro.app/Contents/Frameworks/Kiro"), "Kiro"
        )


FAKE_UCI = """
import sys
buf = bytearray(64 << 20)  # 64 MB that the footprint must see
for line in sys.stdin:
    cmd = line.split()
    if not cmd:
        continue
    if cmd[0] == "uci":
        print("id name fake", flush=True)
        print("uciok", flush=True)
    elif cmd[0] == "isready":
        buf[:: 4096] = b"x" * len(buf[:: 4096])
        print("readyok", flush=True)
    elif cmd[0] == "go":
        print("info depth 1 score cp 10 pv e2e4", flush=True)
        print("bestmove e2e4", flush=True)
    elif cmd[0] == "quit":
        break
"""


@unittest.skipUnless(sys.platform == "darwin", "vmmap is macOS-only")
class FootprintTest(unittest.TestCase):
    """M4: the footprint tool drives a UCI engine and reads vmmap and ps."""

    def test_measures_a_fake_engine(self):
        import footprint

        with tempfile.TemporaryDirectory() as tmp:
            script = pathlib.Path(tmp) / "fake_uci.py"
            script.write_text(FAKE_UCI)
            r = footprint.measure([sys.executable, str(script)], {"Hash": "16"}, 200)
        self.assertGreater(r["after_isready"]["bytes"], 64 << 20)
        self.assertGreater(r["max_rss_bytes"], 64 << 20)
        self.assertGreaterEqual(r["rss_samples"], 1)

    def test_parses_vmmap_sizes(self):
        import footprint

        self.assertEqual(footprint.parse_size("4.4G"), int(4.4 * (1 << 30)))
        self.assertEqual(footprint.parse_size("512K"), 512 << 10)
        self.assertIsNone(footprint.parse_size("n/a"))


class MicrobenchFailureTest(unittest.TestCase):
    """A sweep whose engine produced no measurement must not exit 0."""

    def test_failed_measurements_exit_nonzero(self):
        import microbench

        saved = microbench.run
        microbench.run = lambda *a, **k: "engine failed to start"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = pathlib.Path(tmp) / "nn.json"
                rc = microbench.main(
                    ["nn", "--out", str(out), "--batches", "1", "--lc0", "/nonexistent"]
                )
                self.assertEqual(rc, 1)
                self.assertIn("error", json.loads(out.read_text())["metalfish"][0])
                out = pathlib.Path(tmp) / "ab.json"
                rc = microbench.main(
                    ["ab", "--out", str(out), "--threads", "1", "--repeats", "1"]
                )
                self.assertEqual(rc, 1)
        finally:
            microbench.run = saved


class HostLoadSensitivityTest(unittest.TestCase):
    """Pins the host-load sensitivity analysis of protocol amendment 8."""

    def test_flags_games_overlapping_high_load_minutes(self):
        import make_figures as mf

        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write(
                "2026-10-15T10:00:00 1 1 1 ext_cpu=150% top_ext=x\n"  # old format: BST assumed
                "2026-10-15T10:01:00 1 1 1 ext_cpu=450% top_ext=x\n"
                "2026-10-15T12:40:00+0100 1 1 1 ext_cpu=120% top_ext=x\n"
            )
            path = fh.name
        samples = mf.parse_hostload(pathlib.Path(path))
        self.assertEqual([v for _, v in samples], [150, 450, 120])
        games = [
            {
                "start_time": "2026-10-15T09:59:10 +0100",
                "end_time": "2026-10-15T10:00:30 +0100",
            },
            {
                "start_time": "2026-10-15T10:00:40 +0100",
                "end_time": "2026-10-15T10:01:50 +0100",
            },
            {
                "start_time": "2026-10-15T08:00:00 +0100",
                "end_time": "2026-10-15T08:01:00 +0100",
            },
            {"start_time": "", "end_time": ""},
        ]
        # game 0 ends at 10:00:30; the 10:01:00 sample covers (10:00:00, 10:01:00] -> overlaps
        # game 1 contains the 10:01:00 sample; game 2 predates the logger; game 3 has no times
        self.assertEqual(mf.high_load_games(games, samples, threshold=300), {0, 1})
        self.assertEqual(mf.high_load_games(games, samples, threshold=500), set())


class UnpairedGameAccountingTest(unittest.TestCase):
    """Pins the runner's opening accounting and the statistics tool's pairing when unpaired games
    (a partner game lost to an interruption) sit anywhere in a block."""

    def test_block_openings_consumed(self):
        import run_cell as rc

        k = lambda r, w, f: (str(r), w, f)  # noqa: E731
        pair = lambda r, f: [k(r, "A", f), k(r, "B", f)]  # noqa: E731
        trailing = pair(1, "f1") + pair(2, "f2") + [k(3, "A", "f3")]
        self.assertEqual(rc.block_openings_consumed(trailing, 0), 2)  # f3 is replayed
        replayed = trailing + pair(1, "f3") + pair(2, "f4")
        self.assertEqual(rc.block_openings_consumed(replayed, 0), 4)
        skipped = pair(1, "f1") + [k(2, "A", "f2")] + pair(1, "f3")
        self.assertEqual(
            rc.block_openings_consumed(skipped, 0), 3
        )  # f2 counted as used
        self.assertEqual(rc.block_openings_consumed(replayed, 0, 4), 2)
        hist = [{"block": 0, "games_before": 0}, {"block": 1, "games_before": 9}]
        self.assertEqual(rc.block_end_index(hist, 0, 0), 9)
        self.assertIsNone(rc.block_end_index(hist, 1, 9))

    def test_pairs_survive_round_number_collisions(self):
        def g(rnd, white, fen, result="1/2-1/2"):
            black = "B" if white == "A" else "A"
            return {
                "round": str(rnd),
                "white": white,
                "black": black,
                "result": result,
                "termination": "",
                "fen": fen,
                "plycount": "",
                "comment_tail": "",
                "start_time": "",
                "end_time": "",
            }

        # an unpaired game at round 1 of one invocation and the replay of the same opening at
        # round 1 of the next invocation share the key (1, fen): consecutive pairing keeps the replay
        games = [
            g(1, "A", "f1", "1-0"),
            g(1, "A", "f1"),
            g(1, "B", "f1"),
            g(2, "A", "f2"),
            g(2, "B", "f2"),
        ]
        st = penta_stats.compute(games, "A")
        self.assertEqual((st["pairs"], st["unpaired_games"]), (2, 1))
        # interleaved games (concurrency > 1) still pair by key
        inter = [g(1, "A", "f1"), g(2, "A", "f2"), g(1, "B", "f1"), g(2, "B", "f2")]
        st = penta_stats.compute(inter, "A")
        self.assertEqual((st["pairs"], st["unpaired_games"]), (2, 0))


class AcrossMatchContrastTest(unittest.TestCase):
    """Pins rq6d.md of tools/paper/make_figures.py: RQ6d minus RQ1a with the covariance of the
    openings the two cells share (protocol amendment 12(e))."""

    def test_rq6d_minus_rq1a_with_shared_openings(self):
        import make_figures as mf
        import robustness

        a = [0.25, 0.75, 0.5, 1.0, 0.25, 0.5, 0.75, 0.5]
        d = [0.5, 0.75, 0.5, 1.0, 0.25, 0.75]
        rq1a = [{"fen": f"f{k}", "score": s} for k, s in enumerate(a)]
        rq6d = [{"fen": f"f{k}", "score": s} for k, s in enumerate(d)]
        with tempfile.TemporaryDirectory() as tmp:
            mf.write_rq6d_contrast({"RQ1a": rq1a}, pathlib.Path(tmp))
            self.assertFalse((pathlib.Path(tmp) / "rq6d.md").exists())
            mf.write_rq6d_contrast({"RQ1a": rq1a, "RQ6d": rq6d}, pathlib.Path(tmp))
            md = (pathlib.Path(tmp) / "rq6d.md").read_text()
        con = robustness.linear_combination([(rq6d, 1.0), (rq1a, -1.0)])
        self.assertAlmostEqual(
            con["elo"],
            robustness.elo_of_scores(d)["elo"] - robustness.elo_of_scores(a)["elo"],
        )
        self.assertIn(
            f"| RQ6d minus RQ1a (hints and verification off) | {mf.fmt_pm(con['elo'], con['elo_pm'])} | 6 |",
            md,
        )
        # the pair scores move together on the shared openings, so the contrast is tighter than
        # the same two cells on disjoint openings
        apart = [{"fen": f"g{k}", "score": s} for k, s in enumerate(d)]
        indep = robustness.linear_combination([(apart, 1.0), (rq1a, -1.0)])
        self.assertEqual(indep["shared_openings"], [0])
        self.assertLess(con["elo_pm"], indep["elo_pm"])


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


class OracleShardTest(unittest.TestCase):
    """Pins the contiguous shards of tools/paper/oracle_probe.py (protocol amendment 15)."""

    def setUp(self):
        try:
            import oracle_probe  # noqa: F401
        except ImportError as exc:  # python-chess missing
            self.skipTest(f"oracle_probe unavailable: {exc}")
        self.op = sys.modules["oracle_probe"]

    def test_shards_partition_the_sample_in_order(self):
        for n in (600, 7, 0):
            items = list(range(n))
            for k in (1, 3, 20):
                parts = [self.op.shard_slice(items, f"{i}/{k}") for i in range(k)]
                self.assertEqual([x for p in parts for x in p], items)
                sizes = [len(p) for p in parts]
                self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_bad_shard_is_rejected(self):
        for spec in ("3/3", "-1/3", "0/0"):
            with self.assertRaises((ValueError, ZeroDivisionError)):
                self.op.shard_slice([1, 2, 3], spec)


class RunnerCampaignTest(unittest.TestCase):
    """Pins the runner campaign's job plan and the joining of a runner cell's blocks
    (tools/paper/runner_plan.py, tools/paper/fetch_runner_cell.py; protocol amendment 15).
    """

    def setUp(self):
        import fetch_runner_cell
        import runner_plan

        self.plan, self.fetch = runner_plan, fetch_runner_cell

    def test_games_are_interleaved_block_by_block(self):
        spec = {
            "cells": [
                {"id": "X-R", "machine": "runner", "blocks": 3},
                {"id": "Y-R", "machine": "runner", "blocks": 2},
                {"id": "Z", "blocks": 4},
            ]
        }
        jobs = self.plan.games_matrix(spec, ["X-R", "Y-R"], None)
        self.assertEqual(
            [(j["cell"], j["block"]) for j in jobs],
            [("X-R", 0), ("Y-R", 0), ("X-R", 1), ("Y-R", 1), ("X-R", 2)],
        )
        jobs = self.plan.games_matrix(spec, ["X-R", "Y-R"], [1])
        self.assertEqual(
            [(j["cell"], j["block"]) for j in jobs], [("X-R", 1), ("Y-R", 1)]
        )
        with self.assertRaises(SystemExit):
            self.plan.games_matrix(
                spec, ["Z"], None
            )  # a workstation cell never runs here
        self.assertEqual(
            self.plan.oracle_matrix(4, [0, 3]), [{"shard": 0}, {"shard": 3}]
        )

    def _block(self, root, b, results, sha="aa"):
        d = pathlib.Path(root) / f"games-T-R-b{b}"
        d.mkdir(parents=True)
        games = []
        for i, (r1, r2) in enumerate(results):  # each block's rounds restart at 1
            games.append(pgn_game(i + 1, "AB-T1", "AB-T3", r1))
            games.append(pgn_game(i + 1, "AB-T3", "AB-T1", r2))
        (d / "games.pgn").write_text("".join(games))
        arms = {"a": {"sha256": sha}, "b": {"sha256": sha}}
        hist = [{"block": b, "games_before": 0, "start_opening": 1 + 2 * b}]
        (d / "manifest.json").write_text(json.dumps({"arms": arms, "history": hist}))
        (d / "fastchess_report.txt").write_text(f"# block {b} (0.50 h, exit 0)\nok\n")
        (d / "runner.json").write_text(json.dumps({"image": f"macos-26 {b}"}))
        return d

    def test_blocks_join_in_order_with_cumulative_history(self):
        cell = {
            "id": "T-R",
            "machine": "runner",
            "pairs": 4,
            "blocks": 2,
            "a": {"engine": "AB", "threads": 1},
            "b": {"engine": "AB", "threads": 3},
        }
        with tempfile.TemporaryDirectory() as d:
            blocks = {
                0: self._block(d, 0, [("1-0", "0-1"), ("1/2-1/2", "1/2-1/2")]),
                1: self._block(d, 1, [("0-1", "1-0"), ("1-0", "1-0")]),
            }
            out = pathlib.Path(d) / "out"
            summary = self.fetch.join_cell("T-R", cell, blocks, out)
            self.assertEqual(summary["stats"]["games"], 8)
            self.assertEqual(summary["stats"]["pairs"], 4)
            m = json.load(open(out / "manifest.json"))
            self.assertEqual([h["games_before"] for h in m["history"]], [0, 4])
            self.assertEqual(
                [r["image"] for r in m["runner_blocks"]], ["macos-26 0", "macos-26 1"]
            )
            self.assertEqual([r["block"] for r in summary["blocks_run"]], [0, 1])
            pgn = (out / "games.pgn").read_text()
            self.assertLess(
                pgn.index('[Result "1-0"]'), pgn.index('[Result "1/2-1/2"]')
            )
            self.assertFalse((out / "engines.log").exists())
            with self.assertRaises(SystemExit):  # a joined cell is written once
                self.fetch.join_cell("T-R", cell, blocks, out)

    def test_incomplete_or_mixed_blocks_are_refused(self):
        cell = {
            "id": "T-R",
            "machine": "runner",
            "pairs": 4,
            "blocks": 2,
            "a": {"engine": "AB", "threads": 1},
            "b": {"engine": "AB", "threads": 3},
        }
        with tempfile.TemporaryDirectory() as d:
            short = {
                0: self._block(d, 0, [("1-0", "0-1"), ("1-0", "0-1")]),
                1: self._block(d, 1, [("1-0", "0-1")]),
            }
            with self.assertRaises(SystemExit):
                self.fetch.join_cell("T-R", cell, short, pathlib.Path(d) / "o1")
        with tempfile.TemporaryDirectory() as d:
            mixed = {
                0: self._block(d, 0, [("1-0", "0-1"), ("1-0", "0-1")], sha="aa"),
                1: self._block(d, 1, [("1-0", "0-1"), ("1-0", "0-1")], sha="bb"),
            }
            with self.assertRaises(SystemExit):
                self.fetch.join_cell("T-R", cell, mixed, pathlib.Path(d) / "o2")


class FakeJudge:
    """Stands in for a UCI judge: scores every position +50 cp for the side to move and
    records each call."""

    def __init__(self):
        self.calls = []

    def analyse(self, board, limit, multipv=None, root_moves=None, game=None):
        import chess
        import chess.engine

        self.calls.append(
            {"fen": board.fen(), "limit": limit, "game": game, "multipv": multipv}
        )
        info = {
            "score": chess.engine.PovScore(chess.engine.Cp(50), board.turn),
            "pv": [next(iter(root_moves or board.legal_moves))],
        }
        if multipv:
            return [
                {
                    "score": chess.engine.PovScore(chess.engine.Cp(10 * i), board.turn),
                    "pv": [m],
                }
                for i, m in enumerate(root_moves)
            ]
        return info


class OracleJudgingTest(unittest.TestCase):
    """Pins the equal-effort scoring and the summary of tools/paper/oracle_probe.py
    (amendment §11.12d/g)."""

    def setUp(self):
        try:
            import chess  # noqa: F401
            import oracle_probe  # noqa: F401
        except ImportError as exc:  # python-chess missing
            self.skipTest(f"oracle_probe unavailable: {exc}")
        self.op = sys.modules["oracle_probe"]
        self.chess = sys.modules["chess"]

    def test_logistic_elo(self):
        self.assertEqual(self.op.logistic_elo(0.0), 0.0)
        self.assertAlmostEqual(self.op.logistic_elo(0.1), 400 * math.log10(0.6 / 0.4))
        self.assertAlmostEqual(self.op.logistic_elo(-0.1), -self.op.logistic_elo(0.1))
        self.assertIsNone(self.op.logistic_elo(0.5))
        self.assertAlmostEqual(self.op.logistic(250.0, 250.0), 1 / (1 + math.exp(-1)))

    def test_per_candidate_scoring_uses_equal_effort_and_fresh_trees(self):
        import chess.engine

        eng = FakeJudge()
        board = self.chess.Board()
        limit = chess.engine.Limit(nodes=30000)
        j = self.op.judge_position(
            eng, board, ["e2e4", "d2d4"], limit, per_candidate=True
        )
        self.assertEqual(len(eng.calls), 3)  # root search + one search per candidate
        self.assertTrue(all(c["limit"] is limit for c in eng.calls))
        self.assertTrue(all(c["multipv"] is None for c in eng.calls))
        games = [c["game"] for c in eng.calls]
        self.assertTrue(all(g is not None for g in games))
        self.assertEqual(
            len({id(g) for g in games}), 3
        )  # ucinewgame before every search
        self.assertEqual(eng.calls[1]["fen"].split()[1], "b")  # searched after the move
        # +50 for the opponent to move after the candidate is -50 for the mover
        self.assertEqual(j["cand_cp"], {"e2e4": -50.0, "d2d4": -50.0})
        self.assertEqual(j["best_cp"], 50.0)

    def test_restricted_search_for_time_limited_judges(self):
        import chess.engine

        eng = FakeJudge()
        j = self.op.judge_position(
            eng, self.chess.Board(), ["e2e4", "d2d4"], chess.engine.Limit(time=1.0)
        )
        self.assertEqual(len(eng.calls), 2)
        self.assertEqual(eng.calls[1]["multipv"], 2)
        self.assertIsNone(eng.calls[0]["game"])
        self.assertEqual(j["cand_cp"], {"e2e4": 0.0, "d2d4": 10.0})

    def test_mating_candidate_scored_without_search(self):
        import chess.engine

        eng = FakeJudge()
        board = self.chess.Board(
            "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
        )
        cp = self.op.cp_after_move(
            eng, board, self.chess.Move.from_uci("d8h4"), chess.engine.Limit(nodes=10)
        )
        self.assertEqual(cp, 30000.0)
        self.assertEqual(eng.calls, [])

    @staticmethod
    def _judge(verdict, loss_ab, loss_mcts, delta=0.0):
        return {
            "verdict": verdict,
            "loss_ab_es": loss_ab,
            "loss_mcts_es": loss_mcts,
            "delta_cp": delta,
        }

    def _rows(self):
        def row(stratum, cluster, sf, lc0):
            return {
                "fen": "f",
                "ab_move": "a",
                "mcts_move": "m",
                "selected": "a",
                "stratum": stratum,
                "cluster": cluster,
                "judges": {"sf": sf, "lc0": lc0},
            }

        J = self._judge
        return [
            row("A", "g1", J("mcts", 0.10, 0.0), J("mcts", 0.04, 0.0)),
            row("A", "g2", J("ab", 0.0, 0.05), J("mcts", 0.02, 0.0)),
            row("B", "g3", J("tie", 0.0, 0.0), J("tie", 0.0, 0.0)),
            row("B", "g3", J("ab", 0.0, 0.03), J("ab", 0.0, 0.01)),
        ]

    def test_summary_reweights_strata_and_bounds_on_consensus(self):
        sampling = {"strata_population": {"A": 300, "B": 100}}
        s = self.op.summarise(self._rows(), ["sf", "lc0"], sampling, n_boot=500, seed=1)
        self.assertEqual(
            (s["positions"], s["games"], s["cluster_unit"]), (4, 3, "game")
        )
        sf = s["judges"]["sf"]["stats"]
        self.assertAlmostEqual(sf["frac_mcts_better"]["estimate"], 0.75 * 0.5)
        self.assertAlmostEqual(sf["frac_mcts_better"]["sample_mean"], 0.25)
        self.assertAlmostEqual(sf["arbiter_upper_bound_es"]["estimate"], 0.75 * 0.05)
        self.assertAlmostEqual(sf["realised_accuracy"]["estimate"], 0.75 * 0.5 + 0.25)
        lo, hi = sf["arbiter_upper_bound_es"]["ci95"]
        self.assertLessEqual(lo, 0.75 * 0.05)
        self.assertGreaterEqual(hi, 0.75 * 0.05)
        # stratum B is one game: resampling games keeps its tie and its AB win together
        self.assertEqual(sf["frac_tie"]["ci95"], [0.125, 0.125])
        elo = s["judges"]["sf"]["projected_elo_of_bound"]["estimate"]
        self.assertAlmostEqual(elo, self.op.logistic_elo(0.0375))
        lc0 = s["judges"]["lc0"]["stats"]
        self.assertAlmostEqual(lc0["frac_mcts_better"]["estimate"], 0.75)
        ag = s["agreement"]
        self.assertAlmostEqual(ag["stats"]["frac_judges_agree"]["estimate"], 0.625)
        self.assertAlmostEqual(ag["stats"]["frac_undecided"]["estimate"], 0.375)
        # only the first position has both judges finding the unselected move better: min bound
        self.assertAlmostEqual(
            ag["stats"]["consensus_bound_es"]["estimate"], 0.75 * 0.02
        )
        self.assertAlmostEqual(ag["cohen_kappa_sample"], 0.4375 / 0.6875)
        self.assertEqual(ag["confusion"]["ab/mcts"], 1)

    def test_cross_judge_rates_and_bounds(self):
        """Amendment §11.14: one judge selects, the other scores, in each direction."""
        sampling = {"strata_population": {"A": 300, "B": 100}}
        s = self.op.summarise(self._rows(), ["sf", "lc0"], sampling, n_boot=500, seed=1)
        x = s["cross_judge"]
        self.assertEqual(list(x), ["sf->lc0", "lc0->sf"])
        sl = x["sf->lc0"]
        # sf flags only the first position, and lc0 also rates its MCTS move better
        self.assertEqual(sl["confirmation_rate"]["sample"], {"flagged": 1, "also": 1})
        self.assertAlmostEqual(sl["confirmation_rate"]["estimate"], 1.0)
        self.assertEqual(sl["confirmation_rate"]["ci95"], [1.0, 1.0])
        # a draw of stratum A without game g1 flags nothing and has no ratio
        self.assertLess(sl["confirmation_rate"]["replicates"], 500)
        self.assertEqual(
            sl["scorer_verdicts_on_flagged"], {"ab": 0, "mcts": 1, "tie": 0}
        )
        # sf picks the MCTS move on the first position only (an exact tie keeps the selection);
        # lc0's losses value that pick at 0.04
        self.assertEqual(sl["picks_not_selected"], 1)
        self.assertAlmostEqual(sl["stats"]["cross_bound_es"]["estimate"], 0.75 * 0.02)
        ls = x["lc0->sf"]
        # lc0 flags both stratum-A positions; sf agrees on one of them
        self.assertEqual(ls["confirmation_rate"]["sample"], {"flagged": 2, "also": 1})
        self.assertAlmostEqual(ls["confirmation_rate"]["estimate"], 0.5)
        self.assertEqual(ls["confirmation_rate"]["replicates"], 500)
        self.assertEqual(
            ls["scorer_verdicts_on_flagged"], {"ab": 1, "mcts": 1, "tie": 0}
        )
        # lc0's picks scored by sf: +0.10 on the first position, -0.05 on the second
        self.assertEqual(ls["picks_not_selected"], 2)
        self.assertAlmostEqual(ls["stats"]["cross_bound_es"]["estimate"], 0.75 * 0.025)
        # under the scorer's losses no other judge's pick beats the scorer's own oracle pick
        for c in x.values():
            own = s["judges"][c["scorer"]]["stats"]["arbiter_upper_bound_es"]
            self.assertLessEqual(
                c["stats"]["cross_bound_es"]["estimate"], own["estimate"]
            )
        elo = ls["projected_elo_of_cross_bound"]
        self.assertAlmostEqual(elo["estimate"], self.op.logistic_elo(0.75 * 0.025))
        self.assertIn("not a prediction of game results", elo["label"])

    def test_tables_carry_the_cross_judge_rows(self):
        import make_figures as mf

        sampling = {"strata_population": {"A": 300, "B": 100}}
        s = self.op.summarise(self._rows(), ["sf", "lc0"], sampling, n_boot=200, seed=1)
        s["method"]["expected_score"] = "logistic, 250 cp per unit of log-odds"
        with tempfile.TemporaryDirectory() as d:
            mf.write_oracle_tables(s, pathlib.Path(d))
            md = (pathlib.Path(d) / "oracle.md").read_text()
            tex = (pathlib.Path(d) / "oracle_cross.tex").read_text().splitlines()
            self.assertTrue((pathlib.Path(d) / "oracle.tex").exists())
        self.assertIn(
            "| sf | lc0 | 1 of 4 | 1.000 [1.000, 1.000] | 0/1/0 | 1 | 0.015 [", md
        )
        self.assertIn("| lc0 | sf | 2 of 4 | 0.500 [", md)
        self.assertIn("not a prediction of game results", md)
        body = [l for l in tex if l.endswith("\\\\") and not l.startswith("Selects")]
        self.assertEqual(len(body), 2)
        self.assertTrue(
            body[0].startswith("sf & lc0 & 1 & 1.000 [1.000, 1.000] & 0.015 [")
        )
        self.assertTrue(body[1].startswith("lc0 & sf & 2 & 0.500 ["))

    def test_weights_fall_back_to_row_shares(self):
        rows = self._rows()
        w = self.op.stratum_weights(rows, {})
        self.assertEqual(w, {"A": 0.5, "B": 0.5})

    def test_summarise_only_rewrites_the_summary_without_engines(self):
        with tempfile.TemporaryDirectory() as d:
            moves = [pathlib.Path(d) / "a.jsonl", pathlib.Path(d) / "b.jsonl"]
            out = pathlib.Path(d) / "oracle.jsonl"
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 20"
            for k, path in enumerate(moves):
                with open(path, "w") as fh:
                    for i in range(3):
                        rec = {
                            "fen": fen,
                            "ab_move": "e2e4",
                            "mcts_move": "d2d4",
                            "selected": "e2e4",
                            "mcts_root_visits": i,
                            "game": i,
                            "source": str(path),
                            "engine": "MF",
                        }
                        fh.write(json.dumps(rec) + "\n")
            with open(out, "w") as fh:
                for r in self._rows():
                    fh.write(json.dumps(r) + "\n")
            rc = self.op.main(
                [
                    *map(str, moves),
                    "--out",
                    str(out),
                    "--judge",
                    "sf=-",
                    "--judge",
                    "lc0=-",
                    "--nodes",
                    "lc0:30000",
                    "--summarise-only",
                    "--bootstrap",
                    "200",
                ]
            )
            self.assertEqual(rc, 0)
            summ = json.loads(pathlib.Path(str(out) + ".summary.json").read_text())
            self.assertEqual(summ["positions"], 4)
            self.assertEqual(summ["judging"]["lc0"]["limit"], "nodes 30000")
            self.assertEqual(
                summ["method"]["expected_score"],
                "logistic, 250 cp per unit of log-odds",
            )
            self.assertEqual(summ["sampling"]["method"], "stratified")
            self.assertEqual(sum(summ["sampling"]["strata_population"].values()), 6)


if __name__ == "__main__":
    unittest.main()
