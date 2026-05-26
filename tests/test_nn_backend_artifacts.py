#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import re
import sys
import tempfile
from contextlib import contextmanager

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import check_nn_backend_artifacts as checker  # noqa: E402
from tools import compare_nn_backend_outputs as comparer  # noqa: E402
from tools import run_nn_backend_probe_suite as probe_suite  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


@contextmanager
def argv(args: list[str]):
    old_argv = sys.argv
    sys.argv = ["check_nn_backend_artifacts.py", *args]
    try:
        yield
    finally:
        sys.argv = old_argv


def write_artifacts(
    root: pathlib.Path,
    *,
    backend_label: str = "Metal (MPSGraph) backend",
    has_wdl: bool = True,
    has_moves_left: bool = True,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    report = root / "parity.md"
    compare = root / "comparison.log"
    probe = root / "probe.log"
    manifest = root / "manifest.json"

    report.write_text(
        f"# MetalFish NN Parity Report\n\n- Backend: {backend_label}\n",
        encoding="utf-8",
    )
    compare.write_text(
        f"backend: {backend_label}\n"
        "    benchmark_warmups: 3\n"
        "    batches: b1=1.0ms checksum=0\n"
        f"backend_after: {backend_label}\n",
        encoding="utf-8",
    )
    probe.write_text(
        json.dumps(
            {
                "backend": "metal",
                "network_info": f"{backend_label}\nDevice: synthetic",
                "format": "attention_body=yes, policy=attention",
                "has_wdl": has_wdl,
                "wdl": [0.1, 0.8, 0.1],
                "has_moves_left": has_moves_left,
                "moves_left": 42.0,
                "policy_top": [
                    {"move": "e2e4", "logit": 1.0},
                    {"move": "d2d4", "logit": 0.9},
                    {"move": "g1f3", "logit": 0.8},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return report, compare, probe, manifest


def write_probe(
    path: pathlib.Path,
    *,
    backend: str,
    label: str,
    fen: str = "8/8/8/8/8/8/8/K6k w - - 0 1",
    top_moves: list[str] | None = None,
    value: float = 0.25,
    policy_shift: float = 0.0,
) -> None:
    path.write_text(
        "info string warmup\n"
        + probe_json(
            backend=backend,
            label=label,
            fen=fen,
            top_moves=top_moves,
            value=value,
            policy_shift=policy_shift,
        )
        + "\n",
        encoding="utf-8",
    )


def probe_json(
    *,
    backend: str,
    label: str,
    fen: str = "8/8/8/8/8/8/8/K6k w - - 0 1",
    top_moves: list[str] | None = None,
    value: float = 0.25,
    policy_shift: float = 0.0,
    has_wdl: bool = True,
    has_moves_left: bool = True,
) -> str:
    moves = top_moves or ["e2e4", "d2d4", "g1f3"]
    policy = [1.0 + policy_shift, 0.5, -0.25, -1.0]
    return json.dumps(
        {
            "fen": fen,
            "backend": backend,
            "network_info": f"{label} synthetic",
            "transform": 0,
            "value": value,
            "has_wdl": has_wdl,
            "wdl": [0.2, 0.7, 0.1],
            "has_moves_left": has_moves_left,
            "moves_left": 12.5,
            "policy_top": [
                {"move": moves[0], "logit": 1.0 + policy_shift},
                {"move": moves[1], "logit": 0.5},
                {"move": moves[2], "logit": -0.25},
            ],
            "policy": policy,
        }
    )


def test_checker_writes_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        report, compare, probe, manifest = write_artifacts(pathlib.Path(tmp))
        with argv(
            [
                "--backend-label",
                "Metal (MPSGraph) backend",
                "--parity-report",
                str(report),
                "--comparison-log",
                str(compare),
                "--probe-log",
                str(probe),
                "--manifest-out",
                str(manifest),
                "--min-policy-top",
                "3",
                "--require-batch-benchmark",
            ]
        ):
            expect("checker success", checker.main() == 0)

        data = json.loads(manifest.read_text(encoding="utf-8"))
        expect("manifest backend", data["backend_label"] == "Metal (MPSGraph) backend")
        expect(
            "manifest warmup", data["benchmark_warmup_line"] == "benchmark_warmups: 3"
        )
        expect("manifest batch", "batches:" in data["batch_line"])
        expect("manifest wdl", data["probe"]["has_wdl"] is True)
        expect("manifest moves-left", data["probe"]["has_moves_left"] is True)
        expect("manifest top count", len(data["probe"]["policy_top"]) == 3)


def test_checker_rejects_missing_wdl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        report, compare, probe, manifest = write_artifacts(
            pathlib.Path(tmp), has_wdl=False
        )
        with argv(
            [
                "--backend-label",
                "Metal (MPSGraph) backend",
                "--parity-report",
                str(report),
                "--comparison-log",
                str(compare),
                "--probe-log",
                str(probe),
                "--manifest-out",
                str(manifest),
                "--min-policy-top",
                "3",
                "--require-batch-benchmark",
            ]
        ):
            try:
                checker.main()
            except RuntimeError as exc:
                expect("wdl error", "did not decode WDL" in str(exc))
                return
    raise AssertionError("expected missing WDL to fail")


def test_backend_output_compare_accepts_close_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        expected = root / "metal.log"
        actual = root / "cuda.log"
        summary = root / "summary.json"
        write_probe(expected, backend="metal", label="Metal (MPSGraph) backend")
        write_probe(
            actual,
            backend="cuda",
            label="CUDA transformer backend",
            value=0.2505,
            policy_shift=0.0005,
        )
        with argv(
            [
                "--expected-log",
                str(expected),
                "--actual-log",
                str(actual),
                "--expected-label",
                "Metal (MPSGraph) backend",
                "--actual-label",
                "CUDA transformer backend",
                "--summary-out",
                str(summary),
                "--require-full-policy",
            ]
        ):
            expect("compare success", comparer.main() == 0)
        data = json.loads(summary.read_text(encoding="utf-8"))
        expect("summary actual backend", data["actual_backend"] == "cuda")
        expect(
            "summary policy delta",
            abs(data["policy_max_delta"] - 0.0005) < 1e-9,
        )
        expect("summary probe count", data["probe_count"] == 1)


def test_backend_output_compare_accepts_probe_suite() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        expected = root / "metal-suite.log"
        actual = root / "cuda-suite.log"
        summary = root / "summary.json"
        expected.write_text(
            "\n".join(
                [
                    "info string probe-suite 1/2 name=startpos",
                    probe_json(
                        backend="metal",
                        label="Metal (MPSGraph) backend",
                        fen="8/8/8/8/8/8/8/K6k w - - 0 1",
                    ),
                    "info string probe-suite 2/2 name=promotion",
                    probe_json(
                        backend="metal",
                        label="Metal (MPSGraph) backend",
                        fen="6bk/P7/8/8/8/8/8/K7 w - - 0 1",
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        actual.write_text(
            "\n".join(
                [
                    "info string probe-suite 1/2 name=startpos",
                    probe_json(
                        backend="cuda",
                        label="CUDA transformer backend",
                        fen="8/8/8/8/8/8/8/K6k w - - 0 1",
                        policy_shift=0.0004,
                    ),
                    "info string probe-suite 2/2 name=promotion",
                    probe_json(
                        backend="cuda",
                        label="CUDA transformer backend",
                        fen="6bk/P7/8/8/8/8/8/K7 w - - 0 1",
                        value=0.2507,
                        policy_shift=0.0007,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        with argv(
            [
                "--expected-log",
                str(expected),
                "--actual-log",
                str(actual),
                "--expected-label",
                "Metal (MPSGraph) backend",
                "--actual-label",
                "CUDA transformer backend",
                "--summary-out",
                str(summary),
                "--require-full-policy",
                "--all-probes",
            ]
        ):
            expect("suite compare success", comparer.main() == 0)
        data = json.loads(summary.read_text(encoding="utf-8"))
        expect("suite probe count", data["probe_count"] == 2)
        expect("suite aggregate value", abs(data["max_value_delta"] - 0.0007) < 1e-9)
        expect(
            "suite aggregate policy",
            abs(data["policy_max_delta"] - 0.0007) < 1e-9,
        )


def test_backend_output_compare_accepts_legacy_scalar_probe_suite() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        expected = root / "metal-legacy-suite.log"
        actual = root / "cuda-legacy-suite.log"
        summary = root / "summary.json"
        expected.write_text(
            probe_json(
                backend="metal",
                label="Metal (MPSGraph) backend",
                has_wdl=False,
                has_moves_left=False,
            )
            + "\n",
            encoding="utf-8",
        )
        actual.write_text(
            probe_json(
                backend="cuda",
                label="CUDA transformer backend",
                value=0.2504,
                policy_shift=0.0004,
                has_wdl=False,
                has_moves_left=False,
            )
            + "\n",
            encoding="utf-8",
        )
        with argv(
            [
                "--expected-log",
                str(expected),
                "--actual-log",
                str(actual),
                "--expected-label",
                "Metal (MPSGraph) backend",
                "--actual-label",
                "CUDA transformer backend",
                "--summary-out",
                str(summary),
                "--require-full-policy",
                "--no-require-wdl",
                "--no-require-moves-left",
            ]
        ):
            expect("legacy compare success", comparer.main() == 0)
        data = json.loads(summary.read_text(encoding="utf-8"))
        expect("legacy wdl skipped", data["wdl_delta"] is None)
        expect("legacy moves-left skipped", data["moves_left_delta"] is None)
        expect("legacy value", abs(data["value_delta"] - 0.0004) < 1e-9)


def test_backend_output_compare_rejects_probe_suite_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        expected = root / "metal-suite.log"
        actual = root / "cuda-suite.log"
        expected.write_text(
            probe_json(
                backend="metal",
                label="Metal (MPSGraph) backend",
                fen="8/8/8/8/8/8/8/K6k w - - 0 1",
            )
            + "\n"
            + probe_json(
                backend="metal",
                label="Metal (MPSGraph) backend",
                fen="6bk/P7/8/8/8/8/8/K7 w - - 0 1",
            )
            + "\n",
            encoding="utf-8",
        )
        actual.write_text(
            probe_json(
                backend="cuda",
                label="CUDA transformer backend",
                fen="8/8/8/8/8/8/8/K6k w - - 0 1",
            )
            + "\n"
            + probe_json(
                backend="cuda",
                label="CUDA transformer backend",
                fen="7k/P7/8/8/8/8/8/K7 w - - 0 1",
            )
            + "\n",
            encoding="utf-8",
        )
        with argv(
            [
                "--expected-log",
                str(expected),
                "--actual-log",
                str(actual),
                "--all-probes",
            ]
        ):
            try:
                comparer.main()
            except RuntimeError as exc:
                expect("suite mismatch error", "probe 2" in str(exc))
                expect("suite fen mismatch", "FEN mismatch" in str(exc))
                return
    raise AssertionError("expected suite FEN drift to fail")


def test_backend_output_compare_rejects_top_move_drift() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        expected = root / "metal.log"
        actual = root / "cuda.log"
        write_probe(expected, backend="metal", label="Metal (MPSGraph) backend")
        write_probe(
            actual,
            backend="cuda",
            label="CUDA transformer backend",
            top_moves=["d2d4", "e2e4", "g1f3"],
        )
        with argv(
            [
                "--expected-log",
                str(expected),
                "--actual-log",
                str(actual),
                "--top-count",
                "2",
            ]
        ):
            try:
                comparer.main()
            except RuntimeError as exc:
                expect("top drift error", "top policy move 0 mismatch" in str(exc))
                return
    raise AssertionError("expected top move drift to fail")


def test_probe_suite_runner_writes_multiple_json_probes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        fake_probe = root / "fake_probe.py"
        fake_probe.write_text(
            """#!/usr/bin/env python3
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)
parser.add_argument("--backend", required=True)
parser.add_argument("--fen", required=True)
parser.add_argument("--moves", default="")
parser.add_argument("--top")
parser.add_argument("--warmup")
parser.add_argument("--iterations")
parser.add_argument("--full-policy", action="store_true")
args = parser.parse_args()
print(json.dumps({
    "fen": args.fen,
    "moves": args.moves,
    "final_fen": args.fen,
    "backend": args.backend,
    "network_info": f"{args.backend} synthetic executor=resolved+graph-replay",
    "transform": 0,
    "value": 0.0,
    "has_wdl": True,
    "wdl": [0.1, 0.8, 0.1],
    "has_moves_left": True,
    "moves_left": 1.0,
    "policy_top": [
        {"move": "e2e4", "logit": 1.0},
        {"move": "d2d4", "logit": 0.5},
        {"move": "g1f3", "logit": 0.25},
    ],
    "policy": [1.0, 0.5, 0.25],
}))
""",
            encoding="utf-8",
        )
        fake_probe.chmod(0o755)
        output = root / "suite.log"
        with argv(
            [
                "--probe",
                str(fake_probe),
                "--weights",
                str(root / "weights.pb"),
                "--backend",
                "cuda",
                "--out",
                str(output),
                "--position",
                "one=8/8/8/8/8/8/8/K6k w - - 0 1",
                "--position",
                "two=6bk/P7/8/8/8/8/8/K7 w - - 0 1",
                "--line",
                "three=8/8/8/8/8/8/4P3/K6k w - - 0 1|e2e4",
                "--backend-label",
                "cuda synthetic",
                "--require-network-info-substring",
                "executor=resolved+graph-replay",
                "--require-wdl",
                "--require-moves-left",
                "--expected-policy-count",
                "3",
                "--full-policy",
            ]
        ):
            expect("probe suite success", probe_suite.main() == 0)

        probes = comparer.load_probe_jsons(output)
        expect("probe suite count", len(probes) == 3)
        expect("probe suite first fen", probes[0]["fen"].startswith("8/8/8"))
        expect("probe suite second fen", probes[1]["fen"].startswith("6bk/P7"))
        expect("probe suite line moves", probes[2]["moves"] == "e2e4")


def test_probe_suite_runner_rejects_semantic_drift() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        fake_probe = root / "fake_probe.py"
        fake_probe.write_text(
            """#!/usr/bin/env python3
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)
parser.add_argument("--backend", required=True)
parser.add_argument("--fen", required=True)
parser.add_argument("--moves", default="")
parser.add_argument("--top")
parser.add_argument("--warmup")
parser.add_argument("--iterations")
parser.add_argument("--full-policy", action="store_true")
args = parser.parse_args()
print(json.dumps({
    "fen": args.fen,
    "moves": args.moves,
    "backend": args.backend,
    "network_info": f"{args.backend} synthetic",
    "has_wdl": False,
    "has_moves_left": True,
    "policy": [1.0],
}))
""",
            encoding="utf-8",
        )
        fake_probe.chmod(0o755)
        output = root / "suite.log"
        with argv(
            [
                "--probe",
                str(fake_probe),
                "--weights",
                str(root / "weights.pb"),
                "--backend",
                "cuda",
                "--out",
                str(output),
                "--position",
                "semantic=8/8/8/8/8/8/8/K6k w - - 0 1",
                "--backend-label",
                "cuda synthetic",
                "--require-wdl",
                "--expected-policy-count",
                "3",
                "--full-policy",
            ]
        ):
            try:
                probe_suite.main()
            except RuntimeError as exc:
                message = str(exc)
                expect("probe suite semantic failure name", "semantic" in message)
                expect("probe suite semantic failure reason", "expected WDL" in message)
                return
    raise AssertionError("expected semantic probe drift to fail")


def test_probe_suite_runner_rejects_network_info_drift() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        fake_probe = root / "fake_probe.py"
        fake_probe.write_text(
            """#!/usr/bin/env python3
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)
parser.add_argument("--backend", required=True)
parser.add_argument("--fen", required=True)
parser.add_argument("--moves", default="")
parser.add_argument("--top")
parser.add_argument("--warmup")
parser.add_argument("--iterations")
parser.add_argument("--full-policy", action="store_true")
args = parser.parse_args()
print(json.dumps({
    "fen": args.fen,
    "moves": args.moves,
    "backend": args.backend,
    "network_info": f"{args.backend} synthetic executor=resolved",
    "has_wdl": True,
    "has_moves_left": True,
    "policy": [1.0, 0.5, 0.25],
}))
""",
            encoding="utf-8",
        )
        fake_probe.chmod(0o755)
        output = root / "suite.log"
        with argv(
            [
                "--probe",
                str(fake_probe),
                "--weights",
                str(root / "weights.pb"),
                "--backend",
                "cuda",
                "--out",
                str(output),
                "--position",
                "executor=8/8/8/8/8/8/8/K6k w - - 0 1",
                "--backend-label",
                "cuda synthetic",
                "--require-network-info-substring",
                "executor=resolved+graph-replay",
                "--require-wdl",
                "--require-moves-left",
                "--expected-policy-count",
                "3",
                "--full-policy",
            ]
        ):
            try:
                probe_suite.main()
            except RuntimeError as exc:
                message = str(exc)
                expect("probe suite network-info failure name", "executor" in message)
                expect(
                    "probe suite network-info failure reason",
                    "expected network_info substring" in message,
                )
                return
    raise AssertionError("expected network-info probe drift to fail")


def test_probe_suite_runner_reports_failing_probe_name() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        fake_probe = root / "fake_probe.py"
        fake_probe.write_text(
            """#!/usr/bin/env python3
import sys

print("synthetic probe failure", file=sys.stderr)
sys.exit(7)
""",
            encoding="utf-8",
        )
        fake_probe.chmod(0o755)
        output = root / "suite.log"
        with argv(
            [
                "--probe",
                str(fake_probe),
                "--weights",
                str(root / "weights.pb"),
                "--backend",
                "cuda",
                "--out",
                str(output),
                "--line",
                "badline=8/8/8/8/8/8/4P3/K6k w - - 0 1|e2e4",
            ]
        ):
            try:
                probe_suite.main()
            except RuntimeError as exc:
                expect("probe suite failure name", "badline" in str(exc))
                expect("probe suite failure code", "exit code 7" in str(exc))
                expect(
                    "probe suite stderr captured",
                    "synthetic probe failure" in output.read_text(encoding="utf-8"),
                )
                return
    raise AssertionError("expected probe suite failure to be reported")


def test_windows_cuda_probe_suite_positions_match_python_default() -> None:
    script = (ROOT / "tools/run_gcp_windows_cuda_runtime_gate.sh").read_text(
        encoding="utf-8"
    )
    marker = "  \\$positions = @(\n"
    start = script.index(marker) + len(marker)
    end = script.index("  \\$outBuilder", start)
    lines = script[start:end].splitlines()
    actual = []
    for line in lines:
        match = re.search(
            r'@\{ name = "([^"]+)"; fen = "([^"]+)"; moves = "([^"]*)" \}',
            line,
        )
        if match:
            actual.append(match.groups())

    expected = [
        (position.name, position.fen, position.moves)
        for position in probe_suite.DEFAULT_POSITIONS
    ]
    expect("windows probe suite position count", len(actual) == len(expected))
    expect("windows probe suite positions", actual == expected)


def main() -> int:
    test_checker_writes_manifest()
    test_checker_rejects_missing_wdl()
    test_backend_output_compare_accepts_close_outputs()
    test_backend_output_compare_accepts_probe_suite()
    test_backend_output_compare_accepts_legacy_scalar_probe_suite()
    test_backend_output_compare_rejects_probe_suite_mismatch()
    test_backend_output_compare_rejects_top_move_drift()
    test_probe_suite_runner_writes_multiple_json_probes()
    test_probe_suite_runner_rejects_semantic_drift()
    test_probe_suite_runner_rejects_network_info_drift()
    test_probe_suite_runner_reports_failing_probe_name()
    test_windows_cuda_probe_suite_positions_match_python_default()
    print("NN backend artifact tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
