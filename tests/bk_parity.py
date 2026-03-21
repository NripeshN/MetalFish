#!/usr/bin/env python3
"""Unified BK parity harness for MetalFish MCTS and Lc0."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import chess

PROJ = pathlib.Path(__file__).resolve().parent.parent
METALFISH = PROJ / "build" / "metalfish"
LC0 = PROJ / "reference" / "lc0" / "build" / "release" / "lc0"
WEIGHTS = PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb"

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


def san_to_uci(fen: str, san: str) -> str:
    board = chess.Board(fen)
    return board.parse_san(san).uci()


@dataclass
class SearchResult:
    bestmove: str
    nodes: int
    nps: int
    nn_evals: int
    elapsed: float


class UCISession:
    def __init__(self, cmd: Sequence[str], name: str):
        self.name = name
        self.proc = subprocess.Popen(
            list(cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.send("uci")
        self.wait_for("uciok", 120)
        self.send("isready")
        self.wait_for("readyok", 120)

    def send(self, cmd: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def wait_for(self, prefix: str, timeout: int) -> str:
        deadline = time.time() + timeout
        assert self.proc.stdout is not None
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                continue
            line = line.strip()
            if line.startswith(prefix):
                return line
        raise TimeoutError(f"{self.name}: timeout waiting for {prefix}")

    def setoption(self, name: str, value: str) -> None:
        self.send(f"setoption name {name} value {value}")

    def warmup(self, mode: str, movetime_ms: int, nodes: int) -> None:
        self.send("position startpos")
        if mode == "nodes":
            self.send(f"go nodes {nodes}")
        else:
            self.send(f"go movetime {movetime_ms}")
        self.wait_for("bestmove", 120)
        self.send("isready")
        self.wait_for("readyok", 120)

    def search(self, fen: str, mode: str, movetime_ms: int, nodes: int) -> SearchResult:
        self.send(f"position fen {fen}")
        if mode == "nodes":
            self.send(f"go nodes {nodes}")
        else:
            self.send(f"go movetime {movetime_ms}")

        bestmove = "0000"
        info_nodes = 0
        info_nps = 0
        info_nn_evals = 0
        t0 = time.time()
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                continue
            line = line.strip()
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) > 1:
                    bestmove = parts[1]
                break
            if line.startswith("info "):
                parts = line.split()
                for i, tok in enumerate(parts):
                    if tok == "nodes" and i + 1 < len(parts):
                        try:
                            info_nodes = int(parts[i + 1])
                        except ValueError:
                            pass
                    if tok == "nps" and i + 1 < len(parts):
                        try:
                            info_nps = int(parts[i + 1])
                        except ValueError:
                            pass
                    if tok == "nn_evals" and i + 1 < len(parts):
                        try:
                            info_nn_evals = int(parts[i + 1])
                        except ValueError:
                            pass
        # Drain post-search stats (MetalFish prints final counters asynchronously).
        self.send("isready")
        while True:
            line = self.proc.stdout.readline()
            if not line:
                continue
            line = line.strip()
            if line.startswith("readyok"):
                break
            if "info string   Nodes:" in line:
                try:
                    info_nodes = int(line.split(":")[-1].strip())
                except ValueError:
                    pass
            if "info string   NPS:" in line:
                try:
                    info_nps = int(line.split(":")[-1].strip())
                except ValueError:
                    pass
            if "info string   NN evals:" in line:
                try:
                    info_nn_evals = int(line.split(":")[-1].strip())
                except ValueError:
                    pass

        return SearchResult(
            bestmove=bestmove,
            nodes=info_nodes,
            nps=info_nps,
            nn_evals=info_nn_evals,
            elapsed=time.time() - t0,
        )

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def setup_metalfish(sess: UCISession, weights: pathlib.Path, threads: int, deterministic: bool) -> None:
    sess.setoption("UseMCTS", "true")
    sess.setoption("NNWeights", str(weights))
    sess.setoption("Threads", str(threads))
    sess.setoption("MCTSParityPreset", "true" if deterministic else "false")
    sess.setoption("MCTSAddDirichletNoise", "false")
    sess.send("isready")
    sess.wait_for("readyok", 120)


def setup_lc0(sess: UCISession, threads: int) -> None:
    sess.setoption("Threads", str(threads))
    sess.setoption("Temperature", "0")
    sess.send("isready")
    sess.wait_for("readyok", 120)


def run_engine(
    name: str,
    sess: UCISession,
    mode: str,
    movetime_ms: int,
    nodes: int,
) -> Dict[str, SearchResult]:
    results: Dict[str, SearchResult] = {}
    passed = 0
    print(f"\n{name}:")
    for fen, expected_sans, bk_id in BK_POSITIONS:
        expected_uci = set()
        for san in expected_sans:
            try:
                expected_uci.add(san_to_uci(fen, san))
            except Exception:
                expected_uci.add(san.lower().replace("+", "").replace("#", ""))

        out = sess.search(fen, mode, movetime_ms, nodes)
        results[bk_id] = out
        ok = out.bestmove in expected_uci
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(
            f"  {bk_id}: {status:4s} {out.bestmove:8s}"
            f" expected={expected_sans} nodes={out.nodes} nps={out.nps}"
            f" nn_evals={out.nn_evals} t={out.elapsed:.2f}s"
        )

    print(f"  Score: {passed}/{len(BK_POSITIONS)}")
    return results


def run_once(args: argparse.Namespace) -> int:
    if not args.weights.exists():
        print(f"ERROR: weights not found: {args.weights}")
        return 2

    mode = "nodes" if args.nodes > 0 else "movetime"
    threads = 1 if args.deterministic else args.threads
    nodes = args.nodes if args.nodes > 0 else 800
    movetime_ms = args.movetime

    sessions: Dict[str, UCISession] = {}
    try:
        if args.engine in ("metalfish", "both"):
            if not args.metalfish.exists():
                print(f"ERROR: metalfish not found: {args.metalfish}")
                return 2
            s = UCISession([str(args.metalfish)], "metalfish")
            setup_metalfish(s, args.weights, threads, args.deterministic)
            s.warmup(mode, min(3000, movetime_ms), min(200, nodes))
            sessions["metalfish"] = s

        if args.engine in ("lc0", "both"):
            if not args.lc0.exists():
                print(f"ERROR: lc0 not found: {args.lc0}")
                return 2
            s = UCISession(
                [str(args.lc0), f"--weights={args.weights}", f"--backend={args.backend}"],
                "lc0",
            )
            setup_lc0(s, threads)
            s.warmup(mode, min(3000, movetime_ms), min(200, nodes))
            sessions["lc0"] = s

        all_results: Dict[str, Dict[str, SearchResult]] = {}
        for name, sess in sessions.items():
            all_results[name] = run_engine(name, sess, mode, movetime_ms, nodes)

        if "metalfish" in all_results and "lc0" in all_results:
            agree = 0
            print("\nAgreement:")
            for _, _, bk_id in BK_POSITIONS:
                m = all_results["metalfish"][bk_id].bestmove
                l = all_results["lc0"][bk_id].bestmove
                ok = m == l
                agree += int(ok)
                print(f"  {bk_id}: {'MATCH' if ok else 'DIFF '} {m} vs {l}")
            print(f"  Bestmove agreement: {agree}/{len(BK_POSITIONS)}")
        return 0
    finally:
        for sess in sessions.values():
            sess.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="BK parity harness")
    parser.add_argument("--engine", choices=["metalfish", "lc0", "both"], default="both")
    parser.add_argument("--movetime", type=int, default=10_000)
    parser.add_argument("--nodes", type=int, default=0, help="If >0, uses go nodes")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--backend", default="metal")
    parser.add_argument("--weights", type=pathlib.Path, default=WEIGHTS)
    parser.add_argument("--metalfish", type=pathlib.Path, default=METALFISH)
    parser.add_argument("--lc0", type=pathlib.Path, default=LC0)
    args = parser.parse_args()

    rc = 0
    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"\n=== Run {i + 1}/{args.repeat} ===")
        rc = run_once(args)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
