#!/usr/bin/env python3
"""Run a compact UCI ponder lifecycle smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument(
        "--ponder-go", default="wtime 60000 btime 60000 winc 1000 binc 1000"
    )
    parser.add_argument("--followup-go", default="movetime 150")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--settle-sec", type=float, default=0.6)
    parser.add_argument(
        "--setoption", action="append", default=[], metavar="NAME=VALUE"
    )
    parser.add_argument("--expect-output", action="append", default=[], metavar="TEXT")
    parser.add_argument("--reject-output", action="append", default=[], metavar="TEXT")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--echo-output", action="store_true")
    return parser.parse_args()


def set_option_command(raw: str) -> str:
    if "=" not in raw:
        raise ValueError(f"setoption must be NAME=VALUE, got: {raw}")
    name, value = raw.split("=", 1)
    return f"setoption name {name.strip()} value {value.strip()}"


class UCI:
    def __init__(self, engine: Path, timeout: float) -> None:
        self.timeout = timeout
        self.output: list[str] = []
        self.lines: queue.Queue[str | None] = queue.Queue()
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
        self.lines.put(None)

    def send(self, command: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def read_until(self, predicate, timeout: float | None = None) -> list[str]:
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while time.monotonic() < deadline:
            if self.proc.poll() is not None and self.lines.empty():
                break
            remaining = max(0.05, deadline - time.monotonic())
            try:
                text = self.lines.get(timeout=remaining)
            except queue.Empty:
                break
            if text is None:
                break
            self.output.append(text)
            if predicate(text):
                return self.output
        tail = "\n".join(self.output[-80:])
        raise TimeoutError(f"Timed out waiting for engine response. Tail:\n{tail}")

    def read_for(self, duration: float) -> list[str]:
        deadline = time.monotonic() + max(0.0, duration)
        captured: list[str] = []
        while time.monotonic() < deadline:
            if self.proc.poll() is not None and self.lines.empty():
                break
            remaining = max(0.01, min(0.1, deadline - time.monotonic()))
            try:
                text = self.lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if text is None:
                break
            self.output.append(text)
            captured.append(text)
        return captured

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


def send_position(uci: UCI, position: str) -> None:
    if (
        position == "startpos"
        or position.startswith("fen ")
        or position.startswith("startpos ")
    ):
        uci.send(f"position {position}")
    else:
        uci.send(f"position fen {position}")


def run_ponder_cycle(
    uci: UCI,
    *,
    position: str,
    ponder_go: str,
    settle_sec: float,
    action: str,
) -> str:
    uci.send("ucinewgame")
    uci.send("isready")
    uci.read_until(lambda line: line == "readyok")
    send_position(uci, position)
    uci.send(f"go ponder {ponder_go}")
    early_lines = uci.read_for(settle_sec)
    early_bestmove = next(
        (line for line in early_lines if line.startswith("bestmove ")), None
    )
    if early_bestmove:
        raise RuntimeError(f"early bestmove before {action}: {early_bestmove}")
    uci.send(action)
    uci.read_until(lambda line: line.startswith("bestmove "))
    return next(line for line in reversed(uci.output) if line.startswith("bestmove "))


def main() -> int:
    args = parse_args()
    engine = Path(args.engine)
    if not engine.exists():
        print(f"Engine not found: {engine}", file=sys.stderr)
        return 2

    uci = UCI(engine, args.timeout)
    returncode: int | None = None
    bestmoves: dict[str, str] = {}
    start = time.monotonic()
    try:
        try:
            uci.send("uci")
            uci.read_until(lambda line: line == "uciok")
            for option in args.setoption:
                uci.send(set_option_command(option))
            uci.send("isready")
            uci.read_until(lambda line: line == "readyok")

            bestmoves["ponderhit"] = run_ponder_cycle(
                uci,
                position=args.position,
                ponder_go=args.ponder_go,
                settle_sec=args.settle_sec,
                action="ponderhit",
            )
            bestmoves["stop"] = run_ponder_cycle(
                uci,
                position=args.position,
                ponder_go=args.ponder_go,
                settle_sec=args.settle_sec,
                action="stop",
            )

            send_position(uci, args.position)
            uci.send(f"go {args.followup_go}")
            uci.read_until(lambda line: line.startswith("bestmove "))
            bestmoves["followup"] = next(
                line for line in reversed(uci.output) if line.startswith("bestmove ")
            )
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    finally:
        returncode = uci.close()

    if returncode not in (0, None):
        tail = "\n".join(uci.output[-80:])
        print(f"Engine exited with status {returncode}. Tail:\n{tail}", file=sys.stderr)
        return 1

    output = "\n".join(uci.output)
    for expected in args.expect_output:
        if expected not in output:
            tail = "\n".join(uci.output[-80:])
            print(
                f"Expected output containing {expected!r}. Tail:\n{tail}",
                file=sys.stderr,
            )
            return 1
    for rejected in args.reject_output:
        if rejected in output:
            tail = "\n".join(uci.output[-80:])
            print(
                f"Rejected output containing {rejected!r}. Tail:\n{tail}",
                file=sys.stderr,
            )
            return 1

    elapsed_sec = time.monotonic() - start
    if args.json_out:
        transcript = "\n".join(uci.output)
        payload = {
            "schema": "metalfish.uci_ponder_smoke_result",
            "schema_version": 1,
            "engine": str(engine),
            "position": args.position,
            "ponder_go": args.ponder_go,
            "followup_go": args.followup_go,
            "setoptions": list(args.setoption),
            "bestmoves": bestmoves,
            "elapsed_sec": round(elapsed_sec, 6),
            "returncode": returncode,
            "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
            "transcript_tail": uci.output[-80:],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if args.echo_output:
        print(output)
    print(
        "ponder_smoke "
        f"ponderhit='{bestmoves['ponderhit']}' "
        f"stop='{bestmoves['stop']}' "
        f"followup='{bestmoves['followup']}'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
