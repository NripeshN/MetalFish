#!/usr/bin/env python3
"""
MetalFish Testing Framework
Based on Stockfish's testing.py
"""

import subprocess
from typing import List, Optional
import os
import time
import sys
import traceback
import fnmatch
import io
import pathlib
import concurrent.futures

CYAN_COLOR = "\033[36m"
GRAY_COLOR = "\033[2m"
RED_COLOR = "\033[31m"
GREEN_COLOR = "\033[32m"
RESET_COLOR = "\033[0m"
WHITE_BOLD = "\033[1m"

MAX_TIMEOUT = 60 * 5

PATH = pathlib.Path(__file__).parent.resolve()
BUILD_PATH = PATH.parent / "build"
METALFISH_BIN = BUILD_PATH / "metalfish"


class TimeoutException(Exception):
    def __init__(self, message: str, timeout: int):
        self.message = message
        self.timeout = timeout


def timeout_decorator(timeout: float):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    raise TimeoutException(
                        f"Function {func.__name__} timed out after {timeout} seconds",
                        timeout,
                    )
            return result
        return wrapper
    return decorator


class MetalFish:
    """Wrapper for interacting with MetalFish engine via UCI"""
    
    def __init__(self, path: str = None, args: List[str] = []):
        self.path = path or str(METALFISH_BIN)
        self.process = None
        self.args = args
        self.output = []
        self.start()

    def start(self):
        self.process = subprocess.Popen(
            [self.path] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

    def send_command(self, command: str):
        if not self.process:
            raise RuntimeError("MetalFish process is not started")
        if self.process.poll() is not None:
            raise RuntimeError("MetalFish process has terminated")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def setoption(self, name: str, value: str):
        self.send_command(f"setoption name {name} value {value}")

    @timeout_decorator(MAX_TIMEOUT)
    def expect(self, expected_output: str):
        """Wait for a line matching the pattern (with wildcards)"""
        for line in self.readline():
            if fnmatch.fnmatch(line, expected_output):
                return line

    @timeout_decorator(MAX_TIMEOUT)
    def equals(self, expected_output: str):
        """Wait for a line exactly matching"""
        for line in self.readline():
            if line == expected_output:
                return line

    @timeout_decorator(MAX_TIMEOUT)
    def contains(self, expected_output: str):
        """Wait for a line containing the string"""
        for line in self.readline():
            if expected_output in line:
                return line

    @timeout_decorator(MAX_TIMEOUT)
    def starts_with(self, expected_output: str):
        """Wait for a line starting with the string"""
        for line in self.readline():
            if line.startswith(expected_output):
                return line

    def readline(self):
        if not self.process:
            raise RuntimeError("MetalFish process is not started")
        while True:
            if self.process.poll() is not None:
                raise RuntimeError("MetalFish process has terminated")
            line = self.process.stdout.readline().strip()
            self.output.append(line)
            yield line

    def clear_output(self):
        self.output = []

    def get_output(self) -> List[str]:
        return self.output

    def quit(self):
        self.send_command("quit")

    def close(self):
        if self.process:
            self.process.stdin.close()
            self.process.stdout.close()
            return self.process.wait()
        return 0


class TestRunner:
    """Simple test runner"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_test(self, name: str, test_func):
        print(f"  Testing {name}... ", end="", flush=True)
        try:
            test_func()
            print(f"{GREEN_COLOR}PASSED{RESET_COLOR}")
            self.passed += 1
        except AssertionError as e:
            print(f"{RED_COLOR}FAILED{RESET_COLOR}")
            self.errors.append((name, str(e)))
            self.failed += 1
        except Exception as e:
            print(f"{RED_COLOR}ERROR: {e}{RESET_COLOR}")
            self.errors.append((name, traceback.format_exc()))
            self.failed += 1

    def summary(self):
        print(f"\n{WHITE_BOLD}Test Summary{RESET_COLOR}")
        print(f"  {GREEN_COLOR}{self.passed} passed{RESET_COLOR}, {RED_COLOR}{self.failed} failed{RESET_COLOR}")
        if self.errors:
            print(f"\n{RED_COLOR}Failures:{RESET_COLOR}")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        return self.failed == 0


# ============================================================================
# PERFT TESTS
# ============================================================================

# Standard perft test positions
PERFT_POSITIONS = [
    # (FEN, depth, expected_nodes)
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609),
    # Kiwipete
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603),
    # Position 3
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1, 14),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2, 191),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238),
    # Position 4
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1, 6),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2, 264),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467),
    # Position 5
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 1, 44),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 2, 1486),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379),
]


def run_perft_test(engine: MetalFish, fen: str, depth: int, expected: int) -> bool:
    """Run a single perft test and return True if passed"""
    engine.send_command(f"position fen {fen}")
    engine.send_command(f"go perft {depth}")
    
    # Wait for "Nodes searched: XXXXX"
    for line in engine.readline():
        if line.startswith("Nodes searched:"):
            nodes = int(line.split(":")[1].strip())
            return nodes == expected
    return False


def test_perft():
    """Run all perft tests"""
    print(f"\n{WHITE_BOLD}Running Perft Tests{RESET_COLOR}")
    
    engine = MetalFish()
    
    # Wait for engine to initialize
    engine.contains("MetalFish")
    
    passed = 0
    failed = 0
    
    for fen, depth, expected in PERFT_POSITIONS:
        short_fen = fen[:30] + "..." if len(fen) > 30 else fen
        print(f"  perft({depth}) {short_fen}: ", end="", flush=True)
        
        engine.clear_output()
        engine.send_command(f"position fen {fen}")
        engine.send_command(f"go perft {depth}")
        
        # Wait for result
        try:
            result_line = engine.contains("Nodes searched:")
            nodes = int(result_line.split(":")[1].strip())
            
            if nodes == expected:
                print(f"{GREEN_COLOR}{nodes} ✓{RESET_COLOR}")
                passed += 1
            else:
                print(f"{RED_COLOR}{nodes} ✗ (expected {expected}){RESET_COLOR}")
                failed += 1
        except Exception as e:
            print(f"{RED_COLOR}ERROR: {e}{RESET_COLOR}")
            failed += 1
    
    engine.quit()
    engine.close()
    
    print(f"\n  Perft: {GREEN_COLOR}{passed} passed{RESET_COLOR}, {RED_COLOR}{failed} failed{RESET_COLOR}")
    return failed == 0


# ============================================================================
# UCI TESTS
# ============================================================================

def test_uci_protocol():
    """Test basic UCI protocol compliance"""
    print(f"\n{WHITE_BOLD}Running UCI Protocol Tests{RESET_COLOR}")
    
    runner = TestRunner()
    
    def test_uci_command():
        engine = MetalFish()
        engine.send_command("uci")
        engine.contains("id name MetalFish")
        engine.equals("uciok")
        engine.quit()
        engine.close()
    
    def test_isready_command():
        engine = MetalFish()
        engine.send_command("uci")
        engine.equals("uciok")
        engine.send_command("isready")
        engine.equals("readyok")
        engine.quit()
        engine.close()
    
    def test_position_startpos():
        engine = MetalFish()
        engine.send_command("uci")
        engine.equals("uciok")
        engine.send_command("position startpos")
        engine.send_command("isready")
        engine.equals("readyok")
        engine.quit()
        engine.close()
    
    def test_position_fen():
        engine = MetalFish()
        engine.send_command("uci")
        engine.equals("uciok")
        engine.send_command("position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        engine.send_command("isready")
        engine.equals("readyok")
        engine.quit()
        engine.close()
    
    def test_go_depth():
        engine = MetalFish()
        engine.send_command("uci")
        engine.equals("uciok")
        engine.send_command("position startpos")
        engine.send_command("go depth 3")
        engine.starts_with("bestmove")
        engine.quit()
        engine.close()
    
    def test_ucinewgame():
        engine = MetalFish()
        engine.send_command("uci")
        engine.equals("uciok")
        engine.send_command("ucinewgame")
        engine.send_command("isready")
        engine.equals("readyok")
        engine.quit()
        engine.close()
    
    runner.run_test("uci command", test_uci_command)
    runner.run_test("isready command", test_isready_command)
    runner.run_test("position startpos", test_position_startpos)
    runner.run_test("position fen", test_position_fen)
    runner.run_test("go depth", test_go_depth)
    runner.run_test("ucinewgame", test_ucinewgame)
    
    return runner.summary()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{WHITE_BOLD}MetalFish Test Suite{RESET_COLOR}")
    print("=" * 50)
    
    if not METALFISH_BIN.exists():
        print(f"{RED_COLOR}Error: MetalFish binary not found at {METALFISH_BIN}{RESET_COLOR}")
        print("Please build MetalFish first: cd build && cmake .. && cmake --build .")
        return 1
    
    all_passed = True
    
    # Run UCI tests
    if not test_uci_protocol():
        all_passed = False
    
    # Run perft tests
    if not test_perft():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print(f"{GREEN_COLOR}All tests passed!{RESET_COLOR}")
        return 0
    else:
        print(f"{RED_COLOR}Some tests failed!{RESET_COLOR}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

