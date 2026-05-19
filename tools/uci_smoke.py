#!/usr/bin/env python3
"""Run a small UCI search and wait for a bestmove."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="engine executable")
    parser.add_argument("--position", default="startpos")
    parser.add_argument("--go", default="depth 3")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--expect-bestmove")
    parser.add_argument(
        "--setoption",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="UCI option to set before the search",
    )
    return parser.parse_args()


class UCI:
    def __init__(self, engine: Path, timeout: float) -> None:
        self.timeout = timeout
        self.output: list[str] = []
        self.proc = subprocess.Popen(
            [str(engine)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def send(self, command: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def read_until(self, predicate) -> list[str]:
        assert self.proc.stdout is not None
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            text = line.rstrip()
            self.output.append(text)
            if predicate(text):
                return self.output
        tail = "\n".join(self.output[-80:])
        raise TimeoutError(f"Timed out waiting for engine response. Tail:\n{tail}")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send("quit")
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()


def set_option_command(raw: str) -> str:
    if "=" not in raw:
        raise ValueError(f"setoption must be NAME=VALUE, got: {raw}")
    name, value = raw.split("=", 1)
    return f"setoption name {name.strip()} value {value.strip()}"


def main() -> int:
    args = parse_args()
    engine = Path(args.engine)
    if not engine.exists():
        print(f"Engine not found: {engine}", file=sys.stderr)
        return 2

    uci = UCI(engine, args.timeout)
    try:
        uci.send("uci")
        uci.read_until(lambda line: line == "uciok")
        for option in args.setoption:
            uci.send(set_option_command(option))
        uci.send("isready")
        uci.read_until(lambda line: line == "readyok")
        if args.position == "startpos" or args.position.startswith("fen "):
            uci.send(f"position {args.position}")
        else:
            uci.send(f"position fen {args.position}")
        uci.send(f"go {args.go}")
        uci.read_until(lambda line: line.startswith("bestmove "))
    finally:
        uci.close()

    best_line = next(
        line for line in reversed(uci.output) if line.startswith("bestmove ")
    )
    bestmove = best_line.split()[1]
    print(best_line)
    if args.expect_bestmove and bestmove != args.expect_bestmove:
        print(
            f"Expected bestmove {args.expect_bestmove}, got {bestmove}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
