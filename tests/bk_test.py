#!/usr/bin/env python3
"""BK test suite: MetalFish-MCTS (24 positions, 10s each).

Keeps the engine process alive across all positions so the transformer
model is loaded and the GPU pipeline is warm for every test position.
"""

import chess
import subprocess
import sys
import time
import pathlib
import threading

PROJ = pathlib.Path(__file__).resolve().parent.parent
ENGINE = PROJ / "build" / "metalfish"
WEIGHTS = PROJ / "networks" / "BT4-1024x15x32h-swa-6147500.pb"

MOVETIME_MS = 10_000  # 10 seconds per position

BK_POSITIONS = [
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
    move = board.parse_san(san)
    return move.uci()


class EngineSession:
    """Persistent UCI engine session."""

    def __init__(self, path, weights):
        self.proc = subprocess.Popen(
            [str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name UseMCTS value true")
        self._send(f"setoption name NNWeights value {weights}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, prefix, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline().strip()
            if line.startswith(prefix):
                return line
        raise TimeoutError(f"Timeout waiting for '{prefix}'")

    def warmup(self, seconds=3):
        """Run a short search on startpos to compile Metal shaders."""
        self._send("position startpos")
        self._send(f"go movetime {seconds * 1000}")
        self._wait_for("bestmove", timeout=seconds + 60)
        self._send("isready")
        self._wait_for("readyok")

    def search(self, fen, movetime_ms):
        """Search a position and return (bestmove, nps, nodes)."""
        self._send(f"position fen {fen}")
        self._send(f"go movetime {movetime_ms}")

        bestmove = None
        nps = 0
        nodes = 0
        while True:
            line = self.proc.stdout.readline().strip()
            if line.startswith("bestmove"):
                bestmove = line.split()[1]
                break
            if "info" in line and "nps" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "nps" and i + 1 < len(parts):
                        try:
                            nps = int(parts[i + 1])
                        except ValueError:
                            pass
                    if p == "nodes" and i + 1 < len(parts):
                        try:
                            nodes = int(parts[i + 1])
                        except ValueError:
                            pass
        return bestmove, nps, nodes

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def run_bk_test():
    if not ENGINE.exists():
        print(f"ERROR: Engine not found at {ENGINE}")
        return 1
    if not WEIGHTS.exists():
        print(f"ERROR: Weights not found at {WEIGHTS}")
        return 1

    print(f"BK Test Suite — MetalFish-MCTS, {MOVETIME_MS // 1000}s/position")
    print("=" * 65)

    eng = EngineSession(ENGINE, WEIGHTS)

    print("  Warming up GPU (5s)...", flush=True)
    eng.warmup(5)
    print("  GPU warm.  Starting test.\n")

    passed = 0
    failed = 0

    for fen, bm_san_list, bk_id in BK_POSITIONS:
        bm_uci = set()
        for san in bm_san_list:
            try:
                bm_uci.add(san_to_uci(fen, san))
            except Exception:
                bm_uci.add(san.lower().replace("+", "").replace("#", ""))

        print(f"  {bk_id}: ", end="", flush=True)
        t0 = time.time()

        try:
            bestmove, nps, nodes = eng.search(fen, MOVETIME_MS)
            elapsed = time.time() - t0

            if bestmove and bestmove in bm_uci:
                print(
                    f"\033[32mPASS\033[0m  {bestmove:<8s}"
                    f"  (expected {bm_san_list})"
                    f"  {nodes:>5d}N  {nps:>4d}nps  {elapsed:.1f}s"
                )
                passed += 1
            else:
                print(
                    f"\033[31mFAIL\033[0m  {bestmove:<8s}"
                    f"  (expected {bm_san_list})"
                    f"  {nodes:>5d}N  {nps:>4d}nps  {elapsed:.1f}s"
                )
                failed += 1
        except Exception as e:
            print(f"\033[31mERROR: {e}\033[0m")
            failed += 1

    eng.quit()

    print("=" * 65)
    print(f"Score: {passed}/24  (target: 17/24 for Lc0 parity)")
    if passed >= 17:
        print("\033[32mPARITY TARGET MET\033[0m")
    else:
        print(f"\033[33mBelow parity target by {17 - passed}\033[0m")
    return 0 if passed >= 17 else 1


if __name__ == "__main__":
    sys.exit(run_bk_test())
