#!/usr/bin/env python3
"""
MetalFish Evaluation Harness

Run structured experiments against the EPD test suite:
  --puzzle      Solve EPD positions, report solve rate per engine
  --agreement   Compare two engines' bestmoves at fixed nodes
  --scaling     Run positions at varying thread counts, report NPS/depth
  --games       Run head-to-head games via direct UCI

Usage:
  python3 tools/run_evaluation.py --puzzle
  python3 tools/run_evaluation.py --agreement
  python3 tools/run_evaluation.py --scaling
  python3 tools/run_evaluation.py --games --games-count 10 --tc 60
"""

import subprocess, sys, os, time, json, argparse, re
from pathlib import Path
from datetime import datetime

try:
    import chess
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-chess", "-q"])
    import chess

# ── Paths ──
DIR = Path(__file__).resolve().parent.parent
METALFISH = str(DIR / "build/metalfish")
STOCKFISH = str(DIR / "reference/stockfish/src/stockfish")
LC0 = str(DIR / "reference/lc0/build/release/lc0")
NNWEIGHTS = str(DIR / "networks/BT4-1024x15x32h-swa-6147500.pb")
DEFAULT_EPD = str(DIR / "tests/suites/bk.epd")
RESULTS_DIR = DIR / "results"

C = "\033[36m"; W = "\033[1;37m"; G = "\033[32m"; R = "\033[31m"
Y = "\033[33m"; D = "\033[2m"; B = "\033[1m"; N = "\033[0m"


# ── UCI Engine ──

class UCIEngine:
    """Manages a UCI engine subprocess."""

    def __init__(self, cmd, name, options=None):
        self.name = name
        self.cmd = cmd
        if not Path(cmd).exists():
            raise FileNotFoundError(f"Engine not found: {cmd}")
        self.proc = subprocess.Popen(
            [cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        for key, val in (options or {}).items():
            self._send(f"setoption name {key} value {val}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _readline(self):
        line = self.proc.stdout.readline().strip()
        if line == "" and self.proc.poll() is not None:
            return "bestmove (none)"
        return line

    def _wait_for(self, token):
        while True:
            line = self._readline()
            if line.startswith(token):
                return line

    def newgame(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def go_movetime(self, position_cmd, time_ms):
        """Send position and search for a fixed time. Returns (bestmove, info_lines)."""
        self._send(position_cmd)
        self._send(f"go movetime {time_ms}")
        return self._collect_bestmove()

    def go_nodes(self, position_cmd, nodes):
        """Send position and search for a fixed node count. Returns (bestmove, info_lines)."""
        self._send(position_cmd)
        self._send(f"go nodes {nodes}")
        return self._collect_bestmove()

    def go_timed(self, position_cmd, wtime, btime, winc, binc):
        """Send position with clock-based time control. Returns (bestmove, info_lines)."""
        self._send(position_cmd)
        self._send(f"go wtime {wtime} btime {btime} winc {winc} binc {binc}")
        return self._collect_bestmove()

    def _collect_bestmove(self):
        info_lines = []
        while True:
            line = self._readline()
            if line.startswith("bestmove"):
                parts = line.split()
                bestmove = parts[1] if len(parts) > 1 else None
                return bestmove, info_lines
            elif line.startswith("info") and "depth" in line:
                info_lines.append(line)

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


# ── EPD Parsing ──

def parse_epd(filepath):
    """Parse an EPD file. Supports BK format: FEN ; bm MOVE ; id NAME"""
    positions = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Find bm (best move) -- could be after ; or in the line
            bm_match = re.search(r'\bbm\s+([^;]+)', line)
            id_match = re.search(r'\bid\s+"?([^";]+)"?', line)
            if not bm_match:
                continue
            bm_str = bm_match.group(1).strip().rstrip('+').strip()
            pos_id = id_match.group(1).strip() if id_match else "unknown"

            # FEN is everything before first ; or before bm
            fen_part = line.split(';')[0].strip()
            if ' bm ' in fen_part:
                fen_part = fen_part.split(' bm ')[0].strip()
            fen_fields = fen_part.split()
            if len(fen_fields) >= 4:
                fen = " ".join(fen_fields[:6]) if len(fen_fields) >= 6 else " ".join(fen_fields[:4]) + " 0 1"
            else:
                continue
            positions.append({"fen": fen, "bm": bm_str, "id": pos_id})
    return positions


def san_to_uci(fen, san_move):
    """Convert a SAN move (e.g. Nf3) to UCI (e.g. g1f3) given a FEN."""
    try:
        board = chess.Board(fen)
        move = board.parse_san(san_move)
        return move.uci()
    except Exception:
        return san_move


def uci_to_san(fen, uci_move):
    """Convert a UCI move to SAN for display."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move)
        return board.san(move)
    except Exception:
        return uci_move


def extract_info(info_lines, key):
    """Extract a value from UCI info lines by key name."""
    if not info_lines:
        return None
    last = info_lines[-1]
    parts = last.split()
    for i, p in enumerate(parts):
        if p == key and i + 1 < len(parts):
            return parts[i + 1]
    return None


# ── Save results ──

def save_results(experiment, data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"eval_{experiment}_{ts}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  {G}Results saved:{N} {path}")
    return path


# ── Experiment: Puzzle Solve Rate ──

def run_puzzle(epd_path, time_ms=10000):
    print(f"\n  {C}{B}=== PUZZLE SOLVE RATE ==={N}")
    print(f"  {D}EPD: {epd_path} | Time: {time_ms}ms per position{N}\n")

    positions = parse_epd(epd_path)
    if not positions:
        print(f"  {R}No positions found in {epd_path}{N}")
        return

    engine_configs = [
        ("MetalFish-AB", METALFISH, {"Threads": "4", "Hash": "256"}),
        ("MetalFish-MCTS", METALFISH, {
            "Threads": "4", "Hash": "256",
            "UseMCTS": "true", "NNWeights": NNWEIGHTS,
        }),
        ("MetalFish-Hybrid", METALFISH, {
            "Threads": "4", "Hash": "256",
            "UseHybridSearch": "true", "NNWeights": NNWEIGHTS,
        }),
        ("Stockfish", STOCKFISH, {"Threads": "4", "Hash": "256"}),
    ]

    results = {"experiment": "puzzle", "time_ms": time_ms, "positions": len(positions), "engines": {}}

    for eng_name, eng_cmd, eng_opts in engine_configs:
        if not Path(eng_cmd).exists():
            print(f"  {Y}Skipping {eng_name} (not found){N}")
            continue

        print(f"  {W}Testing {eng_name}...{N}")
        try:
            engine = UCIEngine(eng_cmd, eng_name, eng_opts)
        except FileNotFoundError:
            print(f"  {Y}Skipping {eng_name} (binary missing){N}")
            continue

        solved = 0
        pos_results = []
        for i, pos in enumerate(positions):
            engine.newgame()
            pos_cmd = f"position fen {pos['fen']}"
            bestmove, info = engine.go_movetime(pos_cmd, time_ms)

            expected_uci = san_to_uci(pos["fen"], pos["bm"])
            got_san = uci_to_san(pos["fen"], bestmove) if bestmove else "(none)"
            match = (bestmove == expected_uci)
            if match:
                solved += 1

            depth = extract_info(info, "depth") or "?"
            mark = f"{G}OK{N}" if match else f"{R}MISS{N}"
            print(f"    [{mark}] {pos['id']:30s}  expected={pos['bm']:6s}  got={got_san:6s}  d={depth}")

            pos_results.append({
                "id": pos["id"], "fen": pos["fen"],
                "expected": pos["bm"], "got": got_san,
                "got_uci": bestmove, "solved": match,
                "depth": depth,
            })

        engine.quit()
        rate = solved / len(positions) * 100
        print(f"  {B}{eng_name}: {solved}/{len(positions)} ({rate:.1f}%){N}\n")
        results["engines"][eng_name] = {
            "solved": solved, "total": len(positions),
            "rate_pct": round(rate, 1), "positions": pos_results,
        }

    save_results("puzzle", results)


# ── Experiment: Agreement ──

def run_agreement(epd_path, nodes=500):
    print(f"\n  {C}{B}=== MOVE AGREEMENT ==={N}")
    print(f"  {D}EPD: {epd_path} | Nodes: {nodes}{N}\n")

    positions = parse_epd(epd_path)
    if not positions:
        print(f"  {R}No positions found{N}")
        return

    configs = [
        ("MetalFish-MCTS", METALFISH, {
            "Threads": "4", "Hash": "256",
            "UseMCTS": "true", "NNWeights": NNWEIGHTS,
        }),
        ("Lc0", LC0, {"Threads": "1", "WeightsFile": NNWEIGHTS}),
    ]

    engines = []
    for name, cmd, opts in configs:
        if not Path(cmd).exists():
            print(f"  {R}Engine not found: {cmd} ({name}){N}")
            return
        engines.append(UCIEngine(cmd, name, opts))

    agree = 0
    details = []

    for i, pos in enumerate(positions):
        moves = []
        for eng in engines:
            eng.newgame()
            pos_cmd = f"position fen {pos['fen']}"
            bestmove, info = eng.go_nodes(pos_cmd, nodes)
            san = uci_to_san(pos["fen"], bestmove) if bestmove else "(none)"
            moves.append((eng.name, bestmove, san))

        match = moves[0][1] == moves[1][1]
        if match:
            agree += 1
        mark = f"{G}AGREE{N}" if match else f"{Y}DIFFER{N}"
        print(f"    [{mark}] {pos['id']:30s}  {moves[0][0]}={moves[0][2]:6s}  {moves[1][0]}={moves[1][2]:6s}")
        details.append({
            "id": pos["id"], "fen": pos["fen"],
            moves[0][0]: moves[0][2], moves[1][0]: moves[1][2],
            "agree": match,
        })

    for eng in engines:
        eng.quit()

    rate = agree / len(positions) * 100
    print(f"\n  {B}Agreement: {agree}/{len(positions)} ({rate:.1f}%){N}")

    results = {
        "experiment": "agreement", "nodes": nodes,
        "engines": [c[0] for c in configs],
        "total": len(positions), "agreed": agree,
        "rate_pct": round(rate, 1), "details": details,
    }
    save_results("agreement", results)


# ── Experiment: Scaling ──

def run_scaling(epd_path, thread_counts=None):
    if thread_counts is None:
        thread_counts = [1, 2, 4, 8, 12]

    print(f"\n  {C}{B}=== THREAD SCALING ==={N}")
    print(f"  {D}Threads: {thread_counts}{N}\n")

    positions = parse_epd(epd_path)[:10]  # Use first 10 for speed

    engine_defs = [
        ("MetalFish-AB", METALFISH, {}),
        ("MetalFish-Hybrid", METALFISH, {
            "UseHybridSearch": "true", "NNWeights": NNWEIGHTS,
        }),
    ]

    results = {"experiment": "scaling", "thread_counts": thread_counts, "engines": {}}

    for eng_name, eng_cmd, base_opts in engine_defs:
        if not Path(eng_cmd).exists():
            print(f"  {Y}Skipping {eng_name} (not found){N}")
            continue

        print(f"  {W}{eng_name}{N}")
        eng_results = {}

        for threads in thread_counts:
            opts = {**base_opts, "Threads": str(threads), "Hash": "256"}
            try:
                engine = UCIEngine(eng_cmd, eng_name, opts)
            except FileNotFoundError:
                continue

            total_nps = 0
            total_depth = 0
            count = 0

            for pos in positions:
                engine.newgame()
                pos_cmd = f"position fen {pos['fen']}"
                _, info = engine.go_movetime(pos_cmd, 5000)
                nps_str = extract_info(info, "nps")
                depth_str = extract_info(info, "depth")
                if nps_str:
                    total_nps += int(nps_str)
                    count += 1
                if depth_str:
                    total_depth += int(depth_str)

            engine.quit()

            avg_nps = total_nps // count if count else 0
            avg_depth = total_depth / count if count else 0

            if avg_nps > 1_000_000:
                nps_display = f"{avg_nps/1_000_000:.1f}M"
            elif avg_nps > 1_000:
                nps_display = f"{avg_nps/1_000:.0f}K"
            else:
                nps_display = str(avg_nps)

            print(f"    Threads={threads:2d}  NPS={nps_display:>8s}  avg_depth={avg_depth:.1f}")
            eng_results[threads] = {"avg_nps": avg_nps, "avg_depth": round(avg_depth, 1)}

        results["engines"][eng_name] = eng_results
        print()

    save_results("scaling", results)


# ── Experiment: Head-to-Head Games ──

def run_games(game_count=10, tc_base=60, tc_inc=0.1):
    print(f"\n  {C}{B}=== HEAD-TO-HEAD GAMES ==={N}")
    print(f"  {D}Games: {game_count} | TC: {tc_base}+{tc_inc}{N}\n")

    configs = [
        ("MetalFish-AB", METALFISH, {"Threads": "4", "Hash": "256"}),
        ("MetalFish-MCTS", METALFISH, {
            "Threads": "4", "Hash": "256",
            "UseMCTS": "true", "NNWeights": NNWEIGHTS,
        }),
    ]

    for name, cmd, _ in configs:
        if not Path(cmd).exists():
            print(f"  {R}Engine not found: {cmd} ({name}){N}")
            return

    score = {configs[0][0]: 0.0, configs[1][0]: 0.0}
    game_details = []

    for g in range(game_count):
        if g % 2 == 0:
            w_cfg, b_cfg = configs[0], configs[1]
        else:
            w_cfg, b_cfg = configs[1], configs[0]

        eng_w = UCIEngine(w_cfg[1], w_cfg[0], w_cfg[2])
        eng_b = UCIEngine(b_cfg[1], b_cfg[0], b_cfg[2])
        eng_w.newgame()
        eng_b.newgame()

        board = chess.Board()
        moves_uci = []
        wtime = int(tc_base * 1000)
        btime = int(tc_base * 1000)
        winc = int(tc_inc * 1000)
        binc = int(tc_inc * 1000)
        result = "*"

        while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 500:
            pos_cmd = "position startpos moves " + " ".join(moves_uci) if moves_uci else "position startpos"

            if board.turn == chess.WHITE:
                t0 = time.time()
                bestmove, info = eng_w.go_timed(pos_cmd, max(100, wtime), btime, winc, binc)
                elapsed = int((time.time() - t0) * 1000)
                wtime = wtime - elapsed + winc
                if wtime <= 0:
                    result = "0-1"
                    break
            else:
                t0 = time.time()
                bestmove, info = eng_b.go_timed(pos_cmd, wtime, max(100, btime), winc, binc)
                elapsed = int((time.time() - t0) * 1000)
                btime = btime - elapsed + binc
                if btime <= 0:
                    result = "1-0"
                    break

            if not bestmove or bestmove == "(none)":
                break

            try:
                move = chess.Move.from_uci(bestmove)
                if move not in board.legal_moves:
                    break
            except Exception:
                break

            board.push(move)
            moves_uci.append(bestmove)

        if result == "*":
            if board.is_checkmate():
                result = "0-1" if board.turn == chess.WHITE else "1-0"
            else:
                result = "1/2-1/2"

        eng_w.quit()
        eng_b.quit()

        if result == "1-0":
            score[w_cfg[0]] += 1
        elif result == "0-1":
            score[b_cfg[0]] += 1
        else:
            score[w_cfg[0]] += 0.5
            score[b_cfg[0]] += 0.5

        tag = f"{G}1-0{N}" if result == "1-0" else (f"{R}0-1{N}" if result == "0-1" else f"{Y}draw{N}")
        print(f"    Game {g+1:2d}: {w_cfg[0]} vs {b_cfg[0]}  [{tag}]  {len(board.move_stack)} plies")
        game_details.append({
            "game": g + 1, "white": w_cfg[0], "black": b_cfg[0],
            "result": result, "plies": len(board.move_stack),
        })

    names = [configs[0][0], configs[1][0]]
    print(f"\n  {B}Final: {names[0]} {score[names[0]]} - {score[names[1]]} {names[1]}{N}")

    results = {
        "experiment": "games", "tc": f"{tc_base}+{tc_inc}",
        "total_games": game_count, "score": score, "games": game_details,
    }
    save_results("games", results)


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="MetalFish Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python3 tools/run_evaluation.py --puzzle
  python3 tools/run_evaluation.py --puzzle --epd tests/suites/custom.epd --time 5000
  python3 tools/run_evaluation.py --agreement --nodes 1000
  python3 tools/run_evaluation.py --scaling
  python3 tools/run_evaluation.py --games --games-count 20 --tc 120""",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--puzzle", action="store_true", help="Solve EPD positions, report per-engine solve rate")
    mode.add_argument("--agreement", action="store_true", help="Compare two engines' bestmoves at fixed nodes")
    mode.add_argument("--scaling", action="store_true", help="Run at varying thread counts, report NPS/depth")
    mode.add_argument("--games", action="store_true", help="Run head-to-head games via direct UCI")

    parser.add_argument("--epd", type=str, default=DEFAULT_EPD, help="Path to EPD file")
    parser.add_argument("--time", type=int, default=10000, help="Movetime in ms for puzzle mode (default: 10000)")
    parser.add_argument("--nodes", type=int, default=500, help="Node count for agreement mode (default: 500)")
    parser.add_argument("--threads", type=str, default="1,2,4,8,12", help="Comma-separated thread counts for scaling")
    parser.add_argument("--games-count", type=int, default=10, help="Number of games for games mode (default: 10)")
    parser.add_argument("--tc", type=float, default=60, help="Base time in seconds for games mode (default: 60)")

    args = parser.parse_args()

    if args.puzzle:
        run_puzzle(args.epd, args.time)
    elif args.agreement:
        run_agreement(args.epd, args.nodes)
    elif args.scaling:
        thread_list = [int(t) for t in args.threads.split(",")]
        run_scaling(args.epd, thread_list)
    elif args.games:
        run_games(args.games_count, args.tc)


if __name__ == "__main__":
    main()
