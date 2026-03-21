#!/usr/bin/env python3
"""Paper benchmark suite -- collects all data needed for the MetalFish paper."""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, time, os, datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import chess

PROJ = pathlib.Path(__file__).resolve().parent.parent
METALFISH = PROJ / "build" / "metalfish"
LC0 = PROJ / "reference" / "lc0" / "build" / "release" / "lc0"
STOCKFISH = PROJ / "reference" / "stockfish" / "src" / "stockfish"
PATRICIA = PROJ / "reference" / "Patricia" / "engine" / "patricia"
BERSERK = PROJ / "reference" / "berserk" / "src" / "berserk"
WEIGHTS = PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb"
RESULTS_DIR = PROJ / "results"

BK_POSITIONS: List[Tuple[str, List[str], str]] = [
    ("1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - -", ["Qd1+"], "BK.01"),
    ("3r1k2/4npp1/1ppr3p/p6P/P2PPPP1/1NR5/5K2/2R5 w - -", ["d5"], "BK.02"),
    ("2q1rr1k/3bbnnp/p2p1pp1/2pPp3/PpP1P1P1/1P2BNNP/2BQ1PRK/7R b - -", ["f5"], "BK.03"),
    ("rnbqkb1r/p3pppp/1p6/2ppP3/3N4/2P5/PPP1QPPP/R1B1KB1R w KQkq -", ["e6"], "BK.04"),
    ("r1b2rk1/2q1b1pp/p2ppn2/1p6/3QP3/1BN1B3/PPP3PP/R4RK1 w - -", ["Nd5", "a4"], "BK.05"),
    ("2r3k1/pppR1pp1/4p3/4P1P1/5P2/1P4K1/P1P5/8 w - -", ["g6"], "BK.06"),
    ("1nk1r1r1/pp2n1pp/4p3/q2pPp1N/b1pP1P2/B1P2R2/2P1B1PP/R2Q2K1 w - -", ["Nf6"], "BK.07"),
    ("4b3/p3kp2/6p1/3pP2p/2pP1P2/4K1P1/P3N2P/8 w - -", ["f5"], "BK.08"),
    ("2kr1bnr/pbpq4/2n1pp2/3p3p/3P1P1B/2N2N1Q/PPP3PP/2KR1B1R w - -", ["f5"], "BK.09"),
    ("3rr1k1/pp3pp1/1qn2np1/8/3p4/PP1R1P2/2P1NQPP/R1B3K1 b - -", ["Ne5"], "BK.10"),
    ("2r1nrk1/p2q1ppp/bp1p4/n1pPp3/P1P1P3/2PBB1N1/4QPPP/R4RK1 w - -", ["f4"], "BK.11"),
    ("r3r1k1/ppqb1ppp/8/4p1NQ/8/2P5/PP3PPP/R3R1K1 b - -", ["Bf5"], "BK.12"),
    ("r2q1rk1/4bppp/p2p4/2pP4/3pP3/3Q4/PP1B1PPP/R3R1K1 w - -", ["b4"], "BK.13"),
    ("rnb2r1k/pp2p2p/2pp2p1/q2P1p2/8/1Pb2NP1/PB2PPBP/R2Q1RK1 w - -", ["Qd2", "Qe1"], "BK.14"),
    ("2r3k1/1p2q1pp/2b1pr2/p1pp4/6Q1/1P1PP1R1/P1PN2PP/5RK1 w - -", ["Qxg7+"], "BK.15"),
    ("r1bqkb1r/4npp1/p1p4p/1p1pP1B1/8/1B6/PPPN1PPP/R2Q1RK1 w kq -", ["Ne4"], "BK.16"),
    ("r2q1rk1/1ppnbppp/p2p1nb1/3Pp3/2P1P1P1/2N2N1P/PPB1QP2/R1B2RK1 b - -", ["h5"], "BK.17"),
    ("r1bq1rk1/pp2ppbp/2np2p1/2n5/P3PP2/N1P2N2/1PB3PP/R1B1QRK1 b - -", ["Nb3"], "BK.18"),
    ("3rr3/2pq2pk/p2p1pnp/8/2QBPP2/1P6/P5PP/4RRK1 b - -", ["Rxe4"], "BK.19"),
    ("r4k2/pb2bp1r/1p1qp2p/3pNp2/3P1P2/2N3P1/PPP1Q2P/2KRR3 w - -", ["g4"], "BK.20"),
    ("3rn2k/ppb2rpp/2ppqp2/5N2/2P1P3/1P5Q/PB3PPP/3RR1K1 w - -", ["Nh6"], "BK.21"),
    ("2r2rk1/1bqnbpp1/1p1ppn1p/pP6/N1P1P3/P2B1N1P/1B2QPP1/R2R2K1 b - -", ["Bxe4"], "BK.22"),
    ("r1bqk2r/pp2bppp/2p5/3pP3/P2Q1P2/2N1B3/1PP3PP/R4RK1 b kq -", ["f6"], "BK.23"),
    ("r2qnrnk/p2b2b1/1p1p2pp/2pPpp2/1PP1P3/PRNBB3/3QNPPP/5RK1 w - -", ["f4"], "BK.24"),
]

SCALING_POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppppppp/2n2n2/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 6 7",
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 2 9",
]

def san_to_uci(fen: str, san: str) -> str:
    return chess.Board(fen).parse_san(san).uci()

@dataclass
class SearchResult:
    bestmove: str = "0000"
    nodes: int = 0
    nps: int = 0
    nn_evals: int = 0
    depth: int = 0
    elapsed: float = 0.0

class UCIEngine:
    def __init__(self, cmd: Sequence[str], name: str):
        self.name = name
        self.proc = subprocess.Popen(
            list(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.send("uci")
        self.wait_for("uciok", 120)
        self.send("isready")
        self.wait_for("readyok", 120)

    def send(self, cmd: str) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def wait_for(self, prefix: str, timeout: int = 120) -> str:
        deadline = time.time() + timeout
        assert self.proc.stdout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line: continue
            line = line.strip()
            if line.startswith(prefix): return line
        raise TimeoutError(f"{self.name}: timeout waiting for {prefix}")

    def setoption(self, name: str, value: str) -> None:
        self.send(f"setoption name {name} value {value}")

    def search(self, fen: str, movetime_ms: int = 10000) -> SearchResult:
        self.send(f"position fen {fen}")
        self.send(f"go movetime {movetime_ms}")
        r = SearchResult()
        t0 = time.time()
        assert self.proc.stdout
        while True:
            line = self.proc.stdout.readline()
            if not line: continue
            line = line.strip()
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) > 1: r.bestmove = parts[1]
                break
            if line.startswith("info "):
                parts = line.split()
                for i, tok in enumerate(parts):
                    if tok == "nodes" and i+1 < len(parts):
                        try: r.nodes = int(parts[i+1])
                        except ValueError: pass
                    if tok == "nps" and i+1 < len(parts):
                        try: r.nps = int(parts[i+1])
                        except ValueError: pass
                    if tok == "nn_evals" and i+1 < len(parts):
                        try: r.nn_evals = int(parts[i+1])
                        except ValueError: pass
                    if tok == "depth" and i+1 < len(parts):
                        try: r.depth = int(parts[i+1])
                        except ValueError: pass
        self.send("isready")
        while True:
            line = self.proc.stdout.readline()
            if not line: continue
            line = line.strip()
            if line.startswith("readyok"): break
            if "NPS:" in line:
                try: r.nps = int(line.split(":")[-1].strip())
                except ValueError: pass
            if "Nodes:" in line:
                try: r.nodes = int(line.split(":")[-1].strip())
                except ValueError: pass
            if "NN evals:" in line:
                try: r.nn_evals = int(line.split(":")[-1].strip())
                except ValueError: pass
        r.elapsed = time.time() - t0
        return r

    def warmup(self, movetime_ms: int = 3000) -> None:
        self.send("position startpos")
        self.send(f"go movetime {movetime_ms}")
        self.wait_for("bestmove", 120)
        self.send("isready")
        self.wait_for("readyok", 120)

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

# ============================================================================
# Engine configuration and detection
# ============================================================================

@dataclass
class EngineConfig:
    name: str
    path: pathlib.Path
    cmd_args: List[str] = field(default_factory=list)
    uci_options: Dict[str, str] = field(default_factory=dict)
    available: bool = False

def detect_engines(threads: int = 12) -> Dict[str, EngineConfig]:
    engines = {}
    mf = METALFISH
    w = str(WEIGHTS)

    engines["metalfish-ab"] = EngineConfig(
        name="MetalFish-AB", path=mf,
        uci_options={"Threads": str(threads)})

    engines["metalfish-mcts"] = EngineConfig(
        name="MetalFish-MCTS", path=mf,
        uci_options={"UseMCTS": "true", "NNWeights": w, "Threads": "2"})

    engines["metalfish-hybrid"] = EngineConfig(
        name="MetalFish-Hybrid", path=mf,
        uci_options={"UseHybridSearch": "true", "NNWeights": w, "Threads": str(threads)})

    engines["lc0"] = EngineConfig(
        name="Lc0", path=LC0,
        cmd_args=[f"--weights={w}", "--backend=metal"],
        uci_options={"Threads": "1", "Temperature": "0"})

    engines["stockfish"] = EngineConfig(
        name="Stockfish", path=STOCKFISH,
        uci_options={"Threads": "1", "Skill Level": "20"})

    engines["patricia"] = EngineConfig(
        name="Patricia", path=PATRICIA,
        uci_options={"Threads": "1"})

    engines["berserk"] = EngineConfig(
        name="Berserk", path=BERSERK,
        uci_options={"Threads": "1"})

    for eid, cfg in engines.items():
        cfg.available = cfg.path.exists()

    return engines

def start_engine(cfg: EngineConfig) -> UCIEngine:
    cmd = [str(cfg.path)] + cfg.cmd_args
    eng = UCIEngine(cmd, cfg.name)
    for k, v in cfg.uci_options.items():
        eng.setoption(k, v)
    eng.send("isready")
    eng.wait_for("readyok")
    return eng

# ============================================================================
# Experiment 1: Tactical Accuracy (BK Suite)
# ============================================================================

def run_tactical(engines: Dict[str, EngineConfig], movetime_ms: int = 10000,
                 engine_ids: Optional[List[str]] = None) -> dict:
    print("\n" + "="*60)
    print("EXPERIMENT 1: Tactical Accuracy (Bratko-Kopec)")
    print("="*60)

    if engine_ids is None:
        engine_ids = ["metalfish-ab", "metalfish-mcts", "metalfish-hybrid", "lc0", "stockfish"]

    results = {"experiment": "tactical", "movetime_ms": movetime_ms,
               "timestamp": datetime.datetime.now().isoformat(), "engines": {}, "agreement": {}}

    all_bestmoves: Dict[str, Dict[str, str]] = {}

    for eid in engine_ids:
        cfg = engines.get(eid)
        if not cfg or not cfg.available:
            print(f"\n  SKIP {eid}: not available")
            continue

        print(f"\n--- {cfg.name} ---")
        eng = start_engine(cfg)
        try:
            eng.warmup(min(3000, movetime_ms))
        except TimeoutError:
            print(f"  WARNING: warmup timeout, continuing...")

        score = 0
        positions = []
        bestmoves = {}

        for fen, expected_sans, bk_id in BK_POSITIONS:
            expected_uci = set()
            for san in expected_sans:
                try: expected_uci.add(san_to_uci(fen, san))
                except Exception: expected_uci.add(san.lower().replace("+","").replace("#",""))

            r = eng.search(fen, movetime_ms)
            ok = r.bestmove in expected_uci
            score += int(ok)
            bestmoves[bk_id] = r.bestmove

            pos_data = {"id": bk_id, "fen": fen, "expected": expected_sans,
                        "bestmove": r.bestmove, "pass": ok, "nodes": r.nodes,
                        "nps": r.nps, "nn_evals": r.nn_evals, "depth": r.depth,
                        "time_s": round(r.elapsed, 2)}
            positions.append(pos_data)
            status = "PASS" if ok else "FAIL"
            print(f"  {bk_id}: {status:4s} {r.bestmove:8s} exp={expected_sans}"
                  f" n={r.nodes} nps={r.nps} t={r.elapsed:.1f}s")

        total = len(BK_POSITIONS)
        print(f"  Score: {score}/{total} ({100*score/total:.1f}%)")

        avg_nps = sum(p["nps"] for p in positions) // max(1, len(positions))
        results["engines"][eid] = {
            "name": cfg.name, "score": score, "total": total,
            "solve_rate": round(score/total, 3), "avg_nps": avg_nps,
            "positions": positions}
        all_bestmoves[eid] = bestmoves
        eng.close()

    eids = list(all_bestmoves.keys())
    for i in range(len(eids)):
        for j in range(i+1, len(eids)):
            a, b = eids[i], eids[j]
            agree = sum(1 for bk_id in [p[2] for p in BK_POSITIONS]
                        if all_bestmoves[a].get(bk_id) == all_bestmoves[b].get(bk_id))
            key = f"{a}_vs_{b}"
            results["agreement"][key] = {"agree": agree, "total": len(BK_POSITIONS),
                                         "rate": round(agree/len(BK_POSITIONS), 3)}
            print(f"  Agreement {a} vs {b}: {agree}/{len(BK_POSITIONS)}")

    return results

# ============================================================================
# Experiment 2: NPS Throughput Comparison
# ============================================================================

def run_nps(engines: Dict[str, EngineConfig], thread_counts: List[int],
            movetime_ms: int = 10000) -> dict:
    print("\n" + "="*60)
    print("EXPERIMENT 2: NPS Throughput (MetalFish MCTS vs Lc0)")
    print("="*60)

    test_fens = [
        ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("sicilian", "r1b1kbnr/pp3ppp/2n1p3/2ppP3/3P4/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 5"),
        ("middlegame", "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 6 7"),
        ("endgame", "8/5pk1/5p1p/2R5/pp6/1P4PP/P4PK1/2r5 w - - 0 36"),
    ]

    results = {"experiment": "nps_throughput", "movetime_ms": movetime_ms,
               "timestamp": datetime.datetime.now().isoformat(),
               "thread_counts": thread_counts, "engines": {}}

    for eid in ["metalfish-mcts", "lc0"]:
        cfg = engines.get(eid)
        if not cfg or not cfg.available:
            print(f"\n  SKIP {eid}: not available")
            continue

        results["engines"][eid] = {"name": cfg.name, "by_threads": {}}

        for tc in thread_counts:
            print(f"\n--- {cfg.name} @ {tc} threads ---")
            cfg_copy = EngineConfig(
                name=cfg.name, path=cfg.path, cmd_args=list(cfg.cmd_args),
                uci_options={**cfg.uci_options, "Threads": str(tc)}, available=True)
            eng = start_engine(cfg_copy)
            try: eng.warmup(min(3000, movetime_ms))
            except TimeoutError: print("  WARNING: warmup timeout")

            pos_results = []
            for label, fen in test_fens:
                r = eng.search(fen, movetime_ms)
                pos_results.append({"position": label, "fen": fen,
                    "nodes": r.nodes, "nps": r.nps, "nn_evals": r.nn_evals,
                    "depth": r.depth, "time_s": round(r.elapsed, 2)})
                print(f"  {label}: nps={r.nps} nodes={r.nodes} depth={r.depth}")

            avg_nps = sum(p["nps"] for p in pos_results) // max(1, len(pos_results))
            results["engines"][eid]["by_threads"][str(tc)] = {
                "threads": tc, "avg_nps": avg_nps, "positions": pos_results}
            eng.close()

    mf_data = results["engines"].get("metalfish-mcts", {}).get("by_threads", {})
    lc0_data = results["engines"].get("lc0", {}).get("by_threads", {})
    if mf_data and lc0_data:
        print("\n--- Speedup Ratios ---")
        for tc_str in mf_data:
            mf_nps = mf_data[tc_str]["avg_nps"]
            lc0_nps = lc0_data.get(tc_str, {}).get("avg_nps", 1)
            ratio = mf_nps / max(1, lc0_nps)
            print(f"  {tc_str}T: MetalFish {mf_nps} vs Lc0 {lc0_nps} = {ratio:.2f}x")

    return results

# ============================================================================
# Experiment 3: Thread Scaling
# ============================================================================

def run_scaling(engines: Dict[str, EngineConfig], thread_counts: List[int],
                movetime_ms: int = 10000) -> dict:
    print("\n" + "="*60)
    print("EXPERIMENT 3: Thread Scaling")
    print("="*60)

    results = {"experiment": "thread_scaling", "movetime_ms": movetime_ms,
               "timestamp": datetime.datetime.now().isoformat(),
               "thread_counts": thread_counts, "engines": {}}

    for eid in ["metalfish-ab", "metalfish-mcts", "metalfish-hybrid"]:
        cfg = engines.get(eid)
        if not cfg or not cfg.available:
            print(f"\n  SKIP {eid}: not available")
            continue

        results["engines"][eid] = {"name": cfg.name, "by_threads": {}}
        baseline_nps = None

        for tc in thread_counts:
            print(f"\n--- {cfg.name} @ {tc} threads ---")
            cfg_copy = EngineConfig(
                name=cfg.name, path=cfg.path, cmd_args=list(cfg.cmd_args),
                uci_options={**cfg.uci_options, "Threads": str(tc)}, available=True)
            eng = start_engine(cfg_copy)
            try: eng.warmup(min(3000, movetime_ms))
            except TimeoutError: print("  WARNING: warmup timeout")

            total_nps = 0
            total_nodes = 0
            total_depth = 0
            n_pos = len(SCALING_POSITIONS)

            for i, fen in enumerate(SCALING_POSITIONS):
                r = eng.search(fen, movetime_ms)
                total_nps += r.nps
                total_nodes += r.nodes
                total_depth += r.depth
                print(f"  pos {i+1}/{n_pos}: nps={r.nps} nodes={r.nodes} depth={r.depth}")

            avg_nps = total_nps // n_pos
            avg_depth = total_depth / n_pos
            if baseline_nps is None:
                baseline_nps = max(1, avg_nps)
            scaling = avg_nps / baseline_nps

            results["engines"][eid]["by_threads"][str(tc)] = {
                "threads": tc, "avg_nps": avg_nps, "total_nodes": total_nodes,
                "avg_depth": round(avg_depth, 1), "scaling_factor": round(scaling, 2)}
            print(f"  AVG: nps={avg_nps} depth={avg_depth:.1f} scaling={scaling:.2f}x")
            eng.close()

    return results

# ============================================================================
# Experiment 4: Tournament (simplified head-to-head)
# ============================================================================

def play_game(white: UCIEngine, black: UCIEngine, fen: str,
              movetime_ms: int = 5000, max_moves: int = 200) -> str:
    """Play one game, return '1-0', '0-1', or '1/2-1/2'."""
    board = chess.Board(fen)
    for _ in range(max_moves):
        if board.is_game_over():
            result = board.result()
            return result
        eng = white if board.turn == chess.WHITE else black
        eng.send(f"position fen {board.fen()}")
        eng.send(f"go movetime {movetime_ms}")
        try:
            line = eng.wait_for("bestmove", movetime_ms // 1000 + 30)
        except TimeoutError:
            return "0-1" if board.turn == chess.WHITE else "1-0"
        move_str = line.split()[1] if len(line.split()) > 1 else "0000"
        try:
            move = chess.Move.from_uci(move_str)
            if move in board.legal_moves:
                board.push(move)
            else:
                return "0-1" if board.turn == chess.WHITE else "1-0"
        except Exception:
            return "0-1" if board.turn == chess.WHITE else "1-0"
    return "1/2-1/2"

def run_tournament(engines: Dict[str, EngineConfig], games_per_match: int = 10,
                   movetime_ms: int = 5000) -> dict:
    print("\n" + "="*60)
    print("EXPERIMENT 4: Elo Tournament")
    print("="*60)

    matches = [
        ("metalfish-ab", "patricia"),
        ("metalfish-ab", "berserk"),
        ("metalfish-mcts", "lc0"),
        ("metalfish-mcts", "patricia"),
        ("metalfish-hybrid", "patricia"),
        ("metalfish-hybrid", "berserk"),
        ("metalfish-ab", "metalfish-hybrid"),
        ("metalfish-ab", "metalfish-mcts"),
    ]

    results = {"experiment": "tournament", "movetime_ms": movetime_ms,
               "games_per_match": games_per_match,
               "timestamp": datetime.datetime.now().isoformat(), "matches": []}

    startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    for e1_id, e2_id in matches:
        c1, c2 = engines.get(e1_id), engines.get(e2_id)
        if not c1 or not c1.available or not c2 or not c2.available:
            print(f"\n  SKIP {e1_id} vs {e2_id}: engine not available")
            continue

        print(f"\n--- {c1.name} vs {c2.name} ({games_per_match} games) ---")
        wins, draws, losses = 0, 0, 0

        eng1 = start_engine(c1)
        eng2 = start_engine(c2)

        for g in range(games_per_match):
            if g % 2 == 0:
                result = play_game(eng1, eng2, startpos, movetime_ms)
                if result == "1-0": wins += 1
                elif result == "0-1": losses += 1
                else: draws += 1
            else:
                result = play_game(eng2, eng1, startpos, movetime_ms)
                if result == "1-0": losses += 1
                elif result == "0-1": wins += 1
                else: draws += 1
            print(f"  Game {g+1}: {result} (W{wins}-D{draws}-L{losses})")

        eng1.close()
        eng2.close()

        total = wins + draws + losses
        score = wins + draws * 0.5
        pct = score / max(1, total)
        match_data = {"engine1": e1_id, "engine2": e2_id,
                      "name1": c1.name, "name2": c2.name,
                      "wins": wins, "draws": draws, "losses": losses,
                      "score": score, "total": total, "pct": round(pct, 3)}
        results["matches"].append(match_data)
        print(f"  Result: {c1.name} {wins}W-{draws}D-{losses}L ({pct:.1%})")

    return results

# ============================================================================
# Results Summary Generator
# ============================================================================

def generate_summary(all_results: Dict[str, dict]) -> str:
    lines = ["# MetalFish Paper Benchmark Results", "",
             f"Generated: {datetime.datetime.now().isoformat()}", "",
             f"Hardware: Apple Silicon (see sysctl output)", ""]

    tactical = all_results.get("tactical")
    if tactical:
        lines += ["## Table 1: Tactical Accuracy (Bratko-Kopec, 24 positions)", "",
                  "| Engine | Solved | Rate (%) | Avg NPS |",
                  "|--------|--------|----------|---------|"]
        for eid, data in tactical.get("engines", {}).items():
            lines.append(f"| {data['name']} | {data['score']}/24 | "
                         f"{data['solve_rate']*100:.1f} | {data['avg_nps']:,} |")
        lines += ["", "### Per-Position Breakdown", "",
                  "| Position | " + " | ".join(
                      d["name"] for d in tactical.get("engines", {}).values()) + " |",
                  "|----------|" + "|".join(
                      "-------" for _ in tactical.get("engines", {})) + "|"]
        for i, (fen, exp, bk_id) in enumerate(BK_POSITIONS):
            row = f"| {bk_id} ({','.join(exp)}) |"
            for eid, data in tactical.get("engines", {}).items():
                p = data["positions"][i] if i < len(data["positions"]) else {}
                status = "PASS" if p.get("pass") else "FAIL"
                row += f" {status} ({p.get('bestmove','?')}) |"
            lines.append(row)
        lines += ["", "### Agreement Matrix", ""]
        for key, data in tactical.get("agreement", {}).items():
            lines.append(f"- {key}: {data['agree']}/{data['total']} ({data['rate']*100:.1f}%)")
        lines.append("")

    nps = all_results.get("nps_throughput")
    if nps:
        lines += ["## Table 2: NPS Throughput (MetalFish MCTS vs Lc0)", "",
                  "| Engine | Threads | Avg NPS | Speedup |",
                  "|--------|---------|---------|---------|"]
        mf = nps.get("engines", {}).get("metalfish-mcts", {}).get("by_threads", {})
        lc = nps.get("engines", {}).get("lc0", {}).get("by_threads", {})
        for tc_str in sorted(mf.keys(), key=int):
            mf_nps = mf[tc_str]["avg_nps"]
            lc_nps = lc.get(tc_str, {}).get("avg_nps", 0)
            ratio = mf_nps / max(1, lc_nps) if lc_nps else 0
            lines.append(f"| MetalFish-MCTS | {tc_str} | {mf_nps:,} | -- |")
            if lc_nps:
                lines.append(f"| Lc0 | {tc_str} | {lc_nps:,} | {ratio:.2f}x |")
        lines.append("")

    scaling = all_results.get("thread_scaling")
    if scaling:
        lines += ["## Table 3: Thread Scaling", "",
                  "| Engine | 1T NPS | 2T NPS | 4T NPS | 8T NPS | 8T Scaling |",
                  "|--------|--------|--------|--------|--------|------------|"]
        for eid, data in scaling.get("engines", {}).items():
            bt = data.get("by_threads", {})
            nps_vals = [str(bt.get(str(t), {}).get("avg_nps", "N/A")) for t in [1,2,4,8]]
            sc8 = bt.get("8", {}).get("scaling_factor", "N/A")
            lines.append(f"| {data['name']} | {' | '.join(nps_vals)} | {sc8}x |")
        lines.append("")

    tournament = all_results.get("tournament")
    if tournament:
        lines += ["## Table 4: Tournament Results", "",
                  "| Match | W | D | L | Score | Pct |",
                  "|-------|---|---|---|-------|-----|"]
        for m in tournament.get("matches", []):
            lines.append(f"| {m['name1']} vs {m['name2']} | "
                         f"{m['wins']} | {m['draws']} | {m['losses']} | "
                         f"{m['score']}/{m['total']} | {m['pct']*100:.1f}% |")
        lines.append("")

    return "\n".join(lines)

# ============================================================================
# Main CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Paper benchmark suite")
    parser.add_argument("--tactical", action="store_true", help="Run BK tactical accuracy")
    parser.add_argument("--nps", action="store_true", help="Run NPS throughput comparison")
    parser.add_argument("--scaling", action="store_true", help="Run thread scaling")
    parser.add_argument("--tournament", action="store_true", help="Run Elo tournament")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--threads", type=str, default="12", help="Thread counts (comma-sep)")
    parser.add_argument("--movetime", type=int, default=10000, help="Movetime per position (ms)")
    parser.add_argument("--games", type=int, default=10, help="Games per tournament match")
    parser.add_argument("--tc-movetime", type=int, default=5000, help="Movetime for tournament games")
    args = parser.parse_args()

    if args.all:
        args.tactical = args.nps = args.scaling = args.tournament = True
    if not any([args.tactical, args.nps, args.scaling, args.tournament]):
        args.tactical = args.nps = args.scaling = True

    thread_counts = [int(t.strip()) for t in args.threads.split(",")]
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results: Dict[str, dict] = {}

    engines = detect_engines(threads=thread_counts[0] if len(thread_counts) == 1 else 2)
    print("Detected engines:")
    for eid, cfg in engines.items():
        status = "OK" if cfg.available else "MISSING"
        print(f"  {eid}: {status} ({cfg.path})")

    if args.tactical:
        r = run_tactical(engines, args.movetime)
        all_results["tactical"] = r
        with open(RESULTS_DIR / "paper_tactical.json", "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nSaved: results/paper_tactical.json")

    if args.nps:
        r = run_nps(engines, thread_counts, args.movetime)
        all_results["nps_throughput"] = r
        with open(RESULTS_DIR / "paper_nps.json", "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nSaved: results/paper_nps.json")

    if args.scaling:
        r = run_scaling(engines, thread_counts, args.movetime)
        all_results["thread_scaling"] = r
        with open(RESULTS_DIR / "paper_scaling.json", "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nSaved: results/paper_scaling.json")

    if args.tournament:
        r = run_tournament(engines, args.games, args.tc_movetime)
        all_results["tournament"] = r
        with open(RESULTS_DIR / "paper_tournament.json", "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nSaved: results/paper_tournament.json")

    if all_results:
        summary = generate_summary(all_results)
        summary_path = RESULTS_DIR / "paper_summary.md"
        summary_path.write_text(summary)
        print(f"\nSaved: {summary_path}")
        print("\n" + summary)

    return 0

if __name__ == "__main__":
    sys.exit(main())
