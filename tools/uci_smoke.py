#!/usr/bin/env python3
"""Run a small UCI search and wait for a bestmove."""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
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
        "--expect-output",
        action="append",
        default=[],
        metavar="TEXT",
        help="Substring that must appear in combined engine stdout/stderr; may be repeated",
    )
    parser.add_argument(
        "--reject-output",
        action="append",
        default=[],
        metavar="TEXT",
        help="Substring that must not appear in combined engine stdout/stderr; may be repeated",
    )
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
        self.lines: queue.Queue[str] = queue.Queue()
        self.proc = subprocess.Popen(
            [str(engine)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line.rstrip())

    def send(self, command: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def read_until(self, predicate) -> list[str]:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None and self.lines.empty():
                break
            remaining = max(0.05, deadline - time.monotonic())
            try:
                text = self.lines.get(timeout=remaining)
            except queue.Empty:
                break
            self.output.append(text)
            if predicate(text):
                return self.output
        tail = "\n".join(self.output[-80:])
        raise TimeoutError(f"Timed out waiting for engine response. Tail:\n{tail}")

    def close(self) -> int | None:
        if self.proc.poll() is None:
            try:
                self.send("quit")
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.reader.join(timeout=1)
        return self.proc.returncode


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
    returncode: int | None = None
    try:
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
        except (OSError, TimeoutError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    finally:
        returncode = uci.close()

    if returncode not in (0, None):
        tail = "\n".join(uci.output[-80:])
        print(f"Engine exited with status {returncode}. Tail:\n{tail}", file=sys.stderr)
        return 1

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

    output = "\n".join(uci.output)
    for expected in args.expect_output:
        if expected not in output:
            tail = "\n".join(uci.output[-80:])
            print(
                f"Expected engine output containing {expected!r}. Tail:\n{tail}",
                file=sys.stderr,
            )
            return 1
    for rejected in args.reject_output:
        if rejected in output:
            tail = "\n".join(uci.output[-80:])
            print(
                f"Rejected engine output containing {rejected!r}. Tail:\n{tail}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
