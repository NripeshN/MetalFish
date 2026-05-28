#!/usr/bin/env python3
"""Run a small UCI search and wait for a bestmove."""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import re
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
        "--echo-output",
        action="store_true",
        help="Print the captured engine transcript instead of only the bestmove line",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Write a structured JSON record with the search result and transcript hash.",
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


def parse_int_token(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_info_line(line: str) -> dict:
    tokens = line.split()
    result: dict = {"raw": line}
    for key, field in (
        ("depth", "depth"),
        ("seldepth", "seldepth"),
        ("nodes", "nodes"),
        ("nps", "nps"),
        ("time", "time_ms"),
    ):
        if key in tokens:
            index = tokens.index(key)
            if index + 1 < len(tokens):
                parsed = parse_int_token(tokens[index + 1])
                if parsed is not None:
                    result[field] = parsed
    if "score" in tokens:
        index = tokens.index("score")
        if index + 2 < len(tokens):
            score_value = parse_int_token(tokens[index + 2])
            if score_value is not None:
                result["score"] = {"type": tokens[index + 1], "value": score_value}
    if "pv" in tokens:
        index = tokens.index("pv")
        pv: list[str] = []
        for token in tokens[index + 1 :]:
            if token == "string":
                break
            pv.append(token)
        if pv:
            result["pv"] = pv
    return result


def extract_last_search_info(output: list[str]) -> dict | None:
    for line in reversed(output):
        if line.startswith("info ") and (" pv " in f" {line} " or " score " in line):
            parsed = parse_info_line(line)
            if parsed.get("pv") or parsed.get("score"):
                return parsed
    return None


FINAL_METRIC_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)=([^\s]+)")


def parse_metric_value(raw: str) -> int | float | str:
    parsed_int = parse_int_token(raw)
    if parsed_int is not None:
        return parsed_int
    try:
        return float(raw)
    except ValueError:
        return raw


def extract_final_metrics(output: list[str]) -> dict:
    for line in reversed(output):
        marker = "info string Final:"
        if not line.startswith(marker):
            continue
        return {
            key: parse_metric_value(value)
            for key, value in FINAL_METRIC_RE.findall(line[len(marker) :])
        }
    return {}


def write_json_result(
    path: Path,
    *,
    engine: Path,
    position: str,
    go: str,
    options: list[str],
    bestmove: str,
    output: list[str],
    elapsed_sec: float,
    returncode: int | None,
) -> None:
    transcript = "\n".join(output)
    payload = {
        "schema": "metalfish.uci_smoke_result",
        "schema_version": 1,
        "engine": str(engine),
        "position": position,
        "go": go,
        "setoptions": list(options),
        "bestmove": bestmove,
        "elapsed_sec": round(elapsed_sec, 6),
        "returncode": returncode,
        "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        "transcript_tail": output[-80:],
    }
    search_info = extract_last_search_info(output)
    if search_info:
        payload["search_info"] = search_info
    final_metrics = extract_final_metrics(output)
    if final_metrics:
        payload["final_metrics"] = final_metrics
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
            search_start = time.monotonic()
            uci.send(f"go {args.go}")
            uci.read_until(lambda line: line.startswith("bestmove "))
            search_elapsed = time.monotonic() - search_start
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
    if args.json_out:
        write_json_result(
            args.json_out,
            engine=engine,
            position=args.position,
            go=args.go,
            options=args.setoption,
            bestmove=bestmove,
            output=uci.output,
            elapsed_sec=search_elapsed,
            returncode=returncode,
        )
    if args.echo_output:
        print("\n".join(uci.output))
    else:
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
