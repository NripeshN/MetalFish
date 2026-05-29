#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import pathlib
import sys
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import check_cuda_runtime_manifest as runtime_checker  # noqa: E402
from tools import audit_cuda_gcp_resources as gcp_audit  # noqa: E402
from tools import check_nn_backend_artifacts as checker  # noqa: E402
from tools import compare_nn_backend_benchmarks as benchmark_comparer  # noqa: E402
from tools import compare_nn_backend_outputs as comparer  # noqa: E402
from tools import cuda_runtime_manifest_writer as runtime_writer  # noqa: E402
from tools import cuda_runtime_observed as runtime_observed  # noqa: E402
from tools import cuda_runtime_search_contract as search_contract  # noqa: E402
from tools import download_engine_networks as downloader  # noqa: E402
from tools import dispatch_cuda_release_artifacts as cuda_release_dispatch  # noqa: E402
from tools import fetch_cuda_gpu_gate_inputs as cuda_gpu_inputs  # noqa: E402
from tools import fetch_cuda_release_artifacts as cuda_release  # noqa: E402
from tools import fetch_windows_cuda_runtime_inputs as win_cuda_inputs  # noqa: E402
from tools import run_cuda_runtime_gates_direct as direct_runtime  # noqa: E402
from tools import run_nn_backend_probe_suite as probe_suite  # noqa: E402
from tools import write_portable_manifest as portable_manifest  # noqa: E402


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
    batch_policy_shifts: list[float] | None = None,
) -> str:
    moves = top_moves or ["e2e4", "d2d4", "g1f3"]
    policy = [1.0 + policy_shift, 0.5, -0.25, -1.0]
    payload = {
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
    if batch_policy_shifts is not None:
        payload["batch_outputs"] = [
            {
                "index": index,
                "value": value + shift,
                "has_wdl": has_wdl,
                "wdl": [0.2, 0.7, 0.1],
                "has_moves_left": has_moves_left,
                "moves_left": 12.5,
                "policy_top": [
                    {"move": moves[0], "logit": 1.0 + shift},
                    {"move": moves[1], "logit": 0.5},
                    {"move": moves[2], "logit": -0.25},
                ],
                "policy": [1.0 + shift, 0.5, -0.25, -1.0],
            }
            for index, shift in enumerate(batch_policy_shifts)
        ]
    return json.dumps(payload)


def write_benchmark_log(
    path: pathlib.Path,
    *,
    label: str,
    include_graph_reuse: bool = True,
) -> None:
    graph_line = (
        "    graph_reuse_probe: b4 b1 b2 b4 b1 b2 checksum=2\n"
        if include_graph_reuse
        else ""
    )
    path.write_text(
        f"backend: {label}\n"
        "    benchmark_warmups: 3\n"
        "    batches: b1=6.000ms/6.0000ms_eval "
        "b2=9.000ms/4.5000ms_eval "
        "b4=16.000ms/4.0000ms_eval checksum=1\n"
        f"{graph_line}"
        f"backend_after: {label} executor=resolved+graph-replay(captures=1)\n",
        encoding="utf-8",
    )


def write_selected_batch_benchmark_log(
    path: pathlib.Path,
    *,
    label: str,
    b1_eval_ms: float,
    b16_eval_ms: float,
) -> None:
    path.write_text(
        f"backend: {label}\n"
        "    benchmark_warmups: 3\n"
        f"    batches: b1={b1_eval_ms:.3f}ms/{b1_eval_ms:.4f}ms_eval "
        f"b16={b16_eval_ms * 16.0:.3f}ms/{b16_eval_ms:.4f}ms_eval checksum=1\n"
        "    graph_reuse_probe: b16 b16 checksum=2\n"
        f"backend_after: {label} executor=resolved+graph-replay(captures=1)\n",
        encoding="utf-8",
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


def test_backend_output_compare_accepts_batched_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        expected = root / "metal-batch.log"
        actual = root / "cuda-batch.log"
        summary = root / "summary.json"
        expected.write_text(
            probe_json(
                backend="metal",
                label="Metal (MPSGraph) backend",
                batch_policy_shifts=[0.0, 0.001],
            )
            + "\n",
            encoding="utf-8",
        )
        actual.write_text(
            probe_json(
                backend="cuda",
                label="CUDA transformer backend",
                batch_policy_shifts=[0.0004, 0.0017],
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
            ]
        ):
            expect("batch compare success", comparer.main() == 0)
        data = json.loads(summary.read_text(encoding="utf-8"))
        expect("batch output count", data["batch_output_count"] == 2)
        expect("batch aggregate value", abs(data["max_value_delta"] - 0.0007) < 1e-9)
        expect(
            "batch aggregate policy",
            abs(data["policy_max_delta"] - 0.0007) < 1e-9,
        )


def test_backend_benchmark_compare_writes_summary() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        expected = root / "metal.log"
        actual = root / "cuda.log"
        summary = root / "summary.json"
        write_benchmark_log(expected, label="Metal (MPSGraph) backend")
        write_benchmark_log(actual, label="CUDA transformer backend")
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
                "--min-common-batches",
                "3",
                "--require-graph-reuse",
            ]
        ):
            expect("benchmark compare success", benchmark_comparer.main() == 0)

        data = json.loads(summary.read_text(encoding="utf-8"))
        expect("benchmark common count", data["common_batch_count"] == 3)
        expect("benchmark best common", data["best_common_actual"]["batch_size"] == 4)
        expect(
            "benchmark actual label",
            data["actual"]["label"] == "CUDA transformer backend",
        )
        expect(
            "benchmark graph reuse",
            data["actual"]["graph_reuse_batches"] == [4, 1, 2, 4, 1, 2],
        )


def test_backend_benchmark_compare_checks_selected_release_batch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        expected = root / "metal.log"
        actual = root / "cuda.log"
        summary = root / "summary.json"
        write_selected_batch_benchmark_log(
            expected,
            label="Metal (MPSGraph) backend",
            b1_eval_ms=10.0,
            b16_eval_ms=10.0,
        )
        write_selected_batch_benchmark_log(
            actual,
            label="CUDA transformer backend",
            b1_eval_ms=12.0,
            b16_eval_ms=5.0,
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
                "--require-actual-graph-reuse",
                "--max-batch-eval-ms-ratio",
                "16:1.0",
            ]
        ):
            expect("selected batch compare success", benchmark_comparer.main() == 0)

        data = json.loads(summary.read_text(encoding="utf-8"))
        expect(
            "selected limit recorded",
            data["selected_eval_ms_ratio_limits"]["16"] == 1.0,
        )
        expect("diagnostic worst retained", data["worst_eval_ms_ratio"] > 1.0)


def test_backend_benchmark_compare_rejects_selected_release_batch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        expected = root / "metal.log"
        actual = root / "cuda.log"
        write_selected_batch_benchmark_log(
            expected,
            label="Metal (MPSGraph) backend",
            b1_eval_ms=10.0,
            b16_eval_ms=10.0,
        )
        write_selected_batch_benchmark_log(
            actual,
            label="CUDA transformer backend",
            b1_eval_ms=8.0,
            b16_eval_ms=12.0,
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
                "--require-actual-graph-reuse",
                "--max-batch-eval-ms-ratio",
                "16:1.0",
            ]
        ):
            try:
                benchmark_comparer.main()
            except RuntimeError as exc:
                expect("selected batch rejection", "b16 eval-ms ratio" in str(exc))
                return
    raise AssertionError("expected selected release batch ratio rejection")


def test_backend_benchmark_compare_requires_graph_reuse() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        expected = root / "metal.log"
        actual = root / "cuda.log"
        write_benchmark_log(expected, label="Metal (MPSGraph) backend")
        write_benchmark_log(
            actual,
            label="CUDA transformer backend",
            include_graph_reuse=False,
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
                "--require-graph-reuse",
            ]
        ):
            try:
                benchmark_comparer.main()
            except RuntimeError as exc:
                expect("graph reuse error", "missing graph_reuse_probe" in str(exc))
                return
    raise AssertionError("expected missing graph reuse to fail")


def test_backend_benchmark_compare_allows_expected_without_graph_reuse() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        expected = root / "metal.log"
        actual = root / "cuda.log"
        summary = root / "summary.json"
        write_benchmark_log(
            expected,
            label="Metal (MPSGraph) backend",
            include_graph_reuse=False,
        )
        write_benchmark_log(actual, label="CUDA transformer backend")
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
                "--require-actual-graph-reuse",
            ]
        ):
            expect("actual-only graph reuse success", benchmark_comparer.main() == 0)

        data = json.loads(summary.read_text(encoding="utf-8"))
        expect("expected graph reuse absent", data["expected"]["graph_reuse_batches"] == [])
        expect(
            "actual graph reuse present",
            data["actual"]["graph_reuse_batches"] == [4, 1, 2, 4, 1, 2],
        )


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


def test_backend_output_compare_rejects_batched_top_move_drift() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        expected = root / "metal-batch.log"
        actual = root / "cuda-batch.log"
        expected_payload = json.loads(
            probe_json(
                backend="metal",
                label="Metal (MPSGraph) backend",
                batch_policy_shifts=[0.0, 0.0],
            )
        )
        actual_payload = json.loads(
            probe_json(
                backend="cuda",
                label="CUDA transformer backend",
                batch_policy_shifts=[0.0, 0.0],
            )
        )
        actual_payload["batch_outputs"][1]["policy_top"][0]["move"] = "d2d4"
        expected.write_text(json.dumps(expected_payload) + "\n", encoding="utf-8")
        actual.write_text(json.dumps(actual_payload) + "\n", encoding="utf-8")
        with argv(
            [
                "--expected-log",
                str(expected),
                "--actual-log",
                str(actual),
                "--top-count",
                "1",
            ]
        ):
            try:
                comparer.main()
            except RuntimeError as exc:
                message = str(exc)
                expect("batch drift error", "batch_outputs[1]" in message)
                expect("batch top move drift", "top policy move 0 mismatch" in message)
                return
    raise AssertionError("expected batched top move drift to fail")


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
parser.add_argument("--batch-size")
parser.add_argument("--warmup")
parser.add_argument("--iterations")
parser.add_argument("--full-policy", action="store_true")
args = parser.parse_args()
batch_size = int(args.batch_size or 1)
print(json.dumps({
    "fen": args.fen,
    "moves": args.moves,
    "final_fen": args.fen,
    "backend": args.backend,
    "batch_size_seen": args.batch_size,
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
    "batch_outputs": [
        {
            "index": index,
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
        }
        for index in range(batch_size)
    ],
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
                "--batch-size",
                "2",
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
        expect("probe suite forwards batch size", probes[0]["batch_size_seen"] == "2")
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
parser.add_argument("--batch-size")
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
parser.add_argument("--batch-size")
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


def test_windows_cuda_probe_suite_positions_use_python_default() -> None:
    script = (ROOT / "tools/run_gcp_windows_cuda_runtime_gate.sh").read_text(
        encoding="utf-8"
    )
    expect(
        "windows probe suite imports python defaults",
        "from tools.run_nn_backend_probe_suite import DEFAULT_POSITIONS" in script,
    )
    expect(
        "windows probe suite writes positions json", "probe-positions.json" in script
    )
    expect(
        "windows probe suite wraps json array",
        '"positions": [position.__dict__ for position in DEFAULT_POSITIONS]' in script,
    )
    expect(
        "windows probe suite enumerates positions explicitly",
        "System.Collections.Generic.List[object]" in script
        and "positionsDoc.positions" in script,
    )
    expect("windows probe suite reads json", "ConvertFrom-Json" in script)
    expect(
        "windows probe suite does not duplicate position table",
        '@{ name = "startpos"; fen =' not in script,
    )


def test_windows_cuda_runtime_input_helpers_validate_provenance() -> None:
    run = win_cuda_inputs.RunInfo(
        run_id="123",
        workflow_name="Windows CUDA Compile Gate",
        status="completed",
        conclusion="success",
        head_sha="abc",
        url="https://example.invalid/run/123",
    )
    win_cuda_inputs.require_run_provenance(
        run, expected_workflow="Windows CUDA Compile Gate", expected_sha="abc"
    )

    try:
        win_cuda_inputs.require_run_provenance(
            run, expected_workflow="MetalFish CI", expected_sha="abc"
        )
    except ValueError as exc:
        expect("workflow mismatch message", "MetalFish CI" in str(exc))
    else:
        raise AssertionError("expected workflow provenance mismatch")

    try:
        win_cuda_inputs.require_run_provenance(
            run, expected_workflow="Windows CUDA Compile Gate", expected_sha="def"
        )
    except ValueError as exc:
        expect("sha mismatch message", "expected def" in str(exc))
    else:
        raise AssertionError("expected sha provenance mismatch")


def test_windows_cuda_runtime_input_helpers_select_artifacts() -> None:
    artifacts = [
        win_cuda_inputs.ArtifactInfo(
            artifact_id=1,
            name="metalfish-macos-arm64",
            size_in_bytes=100,
            archive_download_url="https://example.invalid/a.zip",
        ),
        win_cuda_inputs.ArtifactInfo(
            artifact_id=2,
            name="windows-cuda-compile-123",
            size_in_bytes=200,
            archive_download_url="https://example.invalid/b.zip",
        ),
    ]
    expect(
        "metal artifact selected by name",
        win_cuda_inputs.select_artifact(
            artifacts, name="metalfish-macos-arm64"
        ).artifact_id
        == 1,
    )
    expect(
        "windows artifact selected by pattern",
        win_cuda_inputs.select_artifact(
            artifacts, pattern="windows-cuda-compile-*"
        ).artifact_id
        == 2,
    )
    try:
        win_cuda_inputs.select_artifact(artifacts, pattern="missing-*")
    except ValueError as exc:
        expect("missing artifact lists available", "windows-cuda-compile-123" in str(exc))
    else:
        raise AssertionError("expected missing artifact failure")


def test_windows_cuda_runtime_input_manifest_records_file_hashes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        payload = pathlib.Path(tmp) / "payload.log"
        payload.write_text("windows cuda input provenance\n", encoding="utf-8")
        record = win_cuda_inputs.file_record(payload)
        expect("windows input file path", record["path"] == str(payload))
        expect("windows input file size", record["size_bytes"] == payload.stat().st_size)
        expect(
            "windows input file sha",
            record["sha256"]
            == hashlib.sha256(payload.read_bytes()).hexdigest(),
        )

    script = (ROOT / "tools/fetch_windows_cuda_runtime_inputs.py").read_text(
        encoding="utf-8"
    )
    for token in (
        '"schema": "metalfish.windows_cuda_runtime_inputs"',
        '"archive": file_record(windows_archive)',
        '"package": file_record(package)',
        '"archive": file_record(metal_archive)',
        '"files": {',
        "spec.metal_manifest_key",
        "metal_search_paths[spec.key]",
        '"env": env',
    ):
        expect(f"windows input manifest records {token}", token in script)


def test_cuda_runtime_input_helpers_validate_complete_zip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        archive = root / "artifact.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("payload.txt", "synthetic")
        expected_size = archive.stat().st_size
        partial = root / "artifact.zip.part"
        partial.write_bytes(archive.read_bytes()[: max(1, expected_size // 2)])

        expect(
            "windows complete zip",
            win_cuda_inputs.complete_zip(archive, expected_size=expected_size),
        )
        expect(
            "linux complete zip",
            cuda_gpu_inputs.complete_zip(archive, expected_size=expected_size),
        )
        expect(
            "windows rejects wrong size",
            not win_cuda_inputs.complete_zip(archive, expected_size=expected_size + 1),
        )
        expect(
            "linux rejects partial",
            not cuda_gpu_inputs.complete_zip(partial, expected_size=expected_size),
        )
        expect(
            "release complete zip",
            cuda_release.complete_zip(archive, expected_size=expected_size),
        )
        expect(
            "release rejects wrong size",
            not cuda_release.complete_zip(archive, expected_size=expected_size + 1),
        )


def test_cuda_release_artifact_download_retries_truncated_zip() -> None:
    valid_zip = io.BytesIO()
    with zipfile.ZipFile(valid_zip, "w") as archive:
        archive.writestr("payload.txt", "synthetic")
    valid_payload = valid_zip.getvalue()
    truncated_payload = valid_payload[: max(1, len(valid_payload) // 2)]
    artifact = cuda_release.ArtifactInfo(
        artifact_id=123,
        name="cuda-release-artifact",
        size_in_bytes=len(valid_payload),
        archive_download_url="https://example.invalid/artifact.zip",
    )
    old_run = cuda_release.subprocess.run
    old_sleep = cuda_release.time.sleep
    calls: list[list[str]] = []

    class FakeProc:
        returncode = 0
        stderr = b""

    def fake_run(cmd, *, cwd, stdout, stderr):
        calls.append(list(cmd))
        stdout.write(truncated_payload if len(calls) == 1 else valid_payload)
        return FakeProc()

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = pathlib.Path(tmp) / "artifact.zip"
        try:
            cuda_release.subprocess.run = fake_run
            cuda_release.time.sleep = lambda _: None
            cuda_release.download_artifact("owner/repo", artifact, archive_path)
        finally:
            cuda_release.subprocess.run = old_run
            cuda_release.time.sleep = old_sleep
        expect("release download retried", len(calls) == 2)
        expect("release archive exists", archive_path.is_file())
        expect("release archive size", archive_path.stat().st_size == len(valid_payload))
        expect(
            "release archive valid",
            cuda_release.complete_zip(archive_path, expected_size=len(valid_payload)),
        )
        expect("release part cleaned", not archive_path.with_name("artifact.zip.part").exists())
        with zipfile.ZipFile(archive_path) as archive:
            expect(
                "release archive payload",
                archive.read("payload.txt").decode("utf-8") == "synthetic",
            )


def test_windows_cuda_runtime_input_helpers_validate_package_commit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        package_dir = root / "windows-package"
        write_release_package_files(
            package_dir,
            package_kind="windows-cuda",
            windows=True,
            source_commit="compile-sha",
        )
        package = root / "metalfish-windows-x86_64-msvc-cuda.zip"
        with zipfile.ZipFile(package, "w") as archive:
            for path in sorted(package_dir.iterdir()):
                archive.write(path, arcname=path.name)
        summary = win_cuda_inputs.validate_package_manifest(
            package,
            expected_source_commit="compile-sha",
        )
        expect("windows runtime package commit", summary["source_commit"] == "compile-sha")

        try:
            win_cuda_inputs.validate_package_manifest(
                package,
                expected_source_commit="other-sha",
            )
        except ValueError as exc:
            expect("windows package commit drift rejected", "source commit" in str(exc))
            return
    raise AssertionError("expected Windows package commit drift to be rejected")


def write_release_package_files(
    root: pathlib.Path,
    *,
    package_kind: str,
    windows: bool,
    source_commit: str = "abc123",
) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    if windows:
        required = [
            "metalfish.exe",
            "metalfish_nn_probe.exe",
            "test_nn_comparison.exe",
            "PORTABLE_ARTIFACT.md",
            "README.md",
            "CHANGELOG.md",
            "LICENSE",
            "cudart64_12.dll",
            "cublas64_12.dll",
            "cublasLt64_12.dll",
            "abseil_dll.dll",
            "libprotobuf.dll",
            "msvcp140.dll",
            "vcruntime140.dll",
            "z.dll",
        ]
        manifest_name = "windows-cuda-package-manifest.json"
    else:
        required = [
            "metalfish",
            "metalfish_nn_probe",
            "test_nn_comparison",
            "PORTABLE_ARTIFACT.md",
            "README.md",
            "CHANGELOG.md",
            "LICENSE",
        ]
        manifest_name = "linux-cuda-package-manifest.json"
    for name in required:
        path = root / name
        path.write_text(f"{name}\n", encoding="utf-8")
        if not windows and name in {
            "metalfish",
            "metalfish_nn_probe",
            "test_nn_comparison",
        }:
            path.chmod(0o755)
    file_records = []
    for name in required:
        path = root / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        file_records.append(
            {
                "name": name,
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "executable": os.access(path, os.X_OK),
            }
        )
    manifest = {
        "schema": "metalfish.portable_artifact",
        "package": {
            "kind": package_kind,
            "name": f"metalfish-{package_kind}",
            "source_commit": source_commit,
        },
        "files": file_records,
    }
    (root / manifest_name).write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root / manifest_name


def metal_log_record(name: str) -> dict:
    return {
        "path": name,
        "size_bytes": 128,
        "sha256": "f" * 64,
    }


def runtime_artifact_records(
    runtime_kind: str,
    *,
    linux_package: pathlib.Path | None = None,
    artifact_root: pathlib.Path | None = None,
) -> dict:
    records = {}
    for name in runtime_checker.REQUIRED_RELEASE_ARTIFACTS[runtime_kind]:
        if artifact_root is None:
            records[name] = metal_log_record(name)
        else:
            path = artifact_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{runtime_kind}:{name}\n", encoding="utf-8")
            records[name] = cuda_release.file_record(path)
    if runtime_kind == "linux-cuda":
        if linux_package is None:
            records["metalfish-linux-x86_64-cuda.tar.gz"] = metal_log_record(
                "metalfish-linux-x86_64-cuda.tar.gz"
            )
        else:
            records[linux_package.name] = cuda_release.file_record(linux_package)
    return records


def runtime_inputs(*, windows_package: pathlib.Path | None = None) -> dict:
    inputs = {
        "require_metal_compare": "1",
        "require_metal_benchmark_compare": "1",
        "require_metal_search_compare": "1",
        "max_cuda_metal_eval_ms_ratio": "1.0",
        "metal_comparison_log": metal_log_record("metal-comparison.log"),
        "metal_probe_suite_log": metal_log_record("metal-bt4.log"),
        "metal_legacy_probe_suite_log": metal_log_record("metal-legacy.log"),
        "metal_mcts_bk07_search_json": metal_log_record("metal-mcts-bk07.json"),
        "metal_mcts_kiwipete_search_json": metal_log_record(
            "metal-mcts-kiwipete.json"
        ),
        "metal_mcts_after_e4_search_json": metal_log_record(
            "metal-mcts-after-e4.json"
        ),
        "metal_hybrid_bk07_search_json": metal_log_record(
            "metal-hybrid-bk07.json"
        ),
        "metal_hybrid_kiwipete_search_json": metal_log_record(
            "metal-hybrid-kiwipete.json"
        ),
        "metal_hybrid_after_e4_search_json": metal_log_record(
            "metal-hybrid-after-e4.json"
        ),
    }
    if windows_package is not None:
        inputs["package"] = {
            "name": windows_package.name,
            "record": cuda_release.file_record(windows_package),
        }
    return inputs


def runtime_policy(
    *,
    graph: str = "1",
    deterministic_attention_softmax: str = "1",
    full_buffer_clear: str = "1",
    profile: str = "0",
    stable_batch: str = "16",
    cublas_workspace: str = "",
) -> dict:
    return {
        "cuda_graph": graph,
        "cuda_graph_execution": graph,
        "cuda_deterministic_attention_softmax": deterministic_attention_softmax,
        "cuda_full_buffer_clear": full_buffer_clear,
        "cuda_profile": profile,
        "cuda_stable_execution_batch_size": stable_batch,
        "cublas_workspace_config": cublas_workspace,
    }


def observed_runtime_facts(
    *,
    runtime_kind: str = "windows-cuda",
    graph_replay: bool = True,
    stable_batch: int = 16,
    deterministic_attention_softmax: bool = True,
    full_buffer_clear: bool = True,
    stable_batch_eval_ms_ratio: float = 0.75,
    worst_eval_ms_ratio: float = 1.25,
    search_status: str = "passed",
    same_bestmove_required: bool = True,
    bestmove_matches: bool = True,
) -> dict:
    backend_after = {
        "backend_is_cuda": True,
        "cuda_graph_effective": graph_replay,
        "cuda_stable_execution_batch_effective": stable_batch,
        "cuda_deterministic_attention_softmax": deterministic_attention_softmax,
        "cuda_full_buffer_clear_effective": full_buffer_clear,
        "executor_graph_replay": graph_replay,
        "graph_replays": 8 if graph_replay else 0,
    }
    search = {
        "present": True,
        "status": search_status,
        "same_bestmove_required": same_bestmove_required,
        "cuda_bestmove": "h5f6",
        "metal_bestmove": "h5f6" if bestmove_matches else "a3b4",
    }
    return {
        "schema_version": 1,
        "runtime_kind": runtime_kind,
        "benchmark_compare": {
            "present": True,
            "cuda_backend_after": backend_after,
            "stable_batch": stable_batch,
            "stable_batch_eval_ms_ratio": stable_batch_eval_ms_ratio,
            "worst_eval_ms_ratio": worst_eval_ms_ratio,
        },
        "search_compare": {
            "mcts_bk07": dict(search),
            "mcts_kiwipete": dict(search),
            "mcts_after_e4": dict(search),
            "hybrid_bk07": dict(search),
            "hybrid_kiwipete": dict(search),
            "hybrid_after_e4": dict(search),
        },
    }


def write_observed_runtime_inputs(
    root: pathlib.Path,
    *,
    runtime_kind: str,
) -> None:
    if runtime_kind == "linux-cuda":
        prefix = root
        benchmark_name = "metal-cuda-nn-benchmark-summary.json"
    else:
        prefix = root / "logs"
        benchmark_name = "metal-windows-cuda-nn-benchmark-summary.json"
    search_paths = search_contract.search_summary_paths(
        root,
        runtime_kind=runtime_kind,
    )
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / benchmark_name).write_text(
        json.dumps(
            {
                "actual": {
                    "backend_after_line": (
                        "backend_after: CUDA transformer backend "
                        "cuda_graph_effective=true, "
                        "cuda_stable_execution_batch_effective=16, "
                        "cuda_deterministic_attention_softmax=true, "
                        "cuda_full_buffer_clear_effective=true, "
                        "executor=resolved+graph-replay"
                        "(captures=1,replays=58,caches=1,primed=1))"
                    ),
                    "best_batch": {"batch_size": 16, "eval_ms": 3.0},
                },
                "expected": {"label": "Metal (MPSGraph) backend"},
                "best_common_actual": {
                    "batch_size": 16,
                    "actual_eval_ms": 3.0,
                    "expected_eval_ms": 12.0,
                    "actual_speedup_vs_expected": 4.0,
                },
                "common_batches": [
                    {
                        "batch_size": 1,
                        "actual_eval_ms": 14.0,
                        "expected_eval_ms": 12.0,
                        "eval_ms_ratio": 1.1666666667,
                        "actual_speedup_vs_expected": 0.8571428571,
                    },
                    {
                        "batch_size": 16,
                        "actual_eval_ms": 3.0,
                        "expected_eval_ms": 12.0,
                        "eval_ms_ratio": 0.25,
                        "actual_speedup_vs_expected": 4.0,
                    },
                ],
                "common_batch_count": 5,
                "worst_eval_ms_ratio": 1.1666666667,
            }
        ),
        encoding="utf-8",
    )
    for path in search_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "require_same_bestmove": True,
                    "actual": {
                        "bestmove": "h5f6",
                        "search_info": {"nodes": 50},
                    },
                    "expected": {
                        "bestmove": "h5f6",
                        "search_info": {"nodes": 50},
                    },
                }
            ),
            encoding="utf-8",
        )


def test_cuda_runtime_search_contract_paths() -> None:
    root = pathlib.Path("/tmp/metalfish-runtime")
    expect(
        "search contract keys",
        search_contract.search_comparison_keys()
        == (
            "mcts_bk07",
            "mcts_kiwipete",
            "mcts_after_e4",
            "hybrid_bk07",
            "hybrid_kiwipete",
            "hybrid_after_e4",
        ),
    )
    expect(
        "search contract metal inputs",
        [spec.metal_input_key for spec in search_contract.SEARCH_COMPARISONS]
        == [
            "metal_mcts_bk07_search_json",
            "metal_mcts_kiwipete_search_json",
            "metal_mcts_after_e4_search_json",
            "metal_hybrid_bk07_search_json",
            "metal_hybrid_kiwipete_search_json",
            "metal_hybrid_after_e4_search_json",
        ],
    )
    expect(
        "search contract metal artifacts",
        [spec.metal_artifact for spec in search_contract.SEARCH_COMPARISONS]
        == [
            "metal-mcts-bk07-search.json",
            "metal-mcts-kiwipete-search.json",
            "metal-mcts-after-e4-search.json",
            "metal-hybrid-bk07-search.json",
            "metal-hybrid-kiwipete-search.json",
            "metal-hybrid-after-e4-search.json",
        ],
    )
    expect(
        "search contract metal env vars",
        [spec.metalfish_env_var for spec in search_contract.SEARCH_COMPARISONS]
        == [
            "METALFISH_METAL_MCTS_BK07_SEARCH_JSON",
            "METALFISH_METAL_MCTS_KIWIPETE_SEARCH_JSON",
            "METALFISH_METAL_MCTS_AFTER_E4_SEARCH_JSON",
            "METALFISH_METAL_HYBRID_BK07_SEARCH_JSON",
            "METALFISH_METAL_HYBRID_KIWIPETE_SEARCH_JSON",
            "METALFISH_METAL_HYBRID_AFTER_E4_SEARCH_JSON",
        ],
    )
    expect(
        "search contract gate env vars",
        [spec.gate_env_var for spec in search_contract.SEARCH_COMPARISONS]
        == [
            "GATE_METAL_MCTS_BK07_SEARCH_JSON",
            "GATE_METAL_MCTS_KIWIPETE_SEARCH_JSON",
            "GATE_METAL_MCTS_AFTER_E4_SEARCH_JSON",
            "GATE_METAL_HYBRID_BK07_SEARCH_JSON",
            "GATE_METAL_HYBRID_KIWIPETE_SEARCH_JSON",
            "GATE_METAL_HYBRID_AFTER_E4_SEARCH_JSON",
        ],
    )
    linux = search_contract.search_summary_paths(root, runtime_kind="linux-cuda")
    windows = search_contract.search_summary_paths(root, runtime_kind="windows-cuda")
    metal = search_contract.metal_artifact_paths(root / "build")
    expect(
        "linux search summary artifact names",
        search_contract.search_summary_artifact_names(runtime_kind="linux-cuda")
        == {
            "metal-cuda-mcts-bk07-search-summary.json",
            "metal-cuda-mcts-kiwipete-search-summary.json",
            "metal-cuda-mcts-after-e4-search-summary.json",
            "metal-cuda-hybrid-bk07-search-summary.json",
            "metal-cuda-hybrid-kiwipete-search-summary.json",
            "metal-cuda-hybrid-after-e4-search-summary.json",
        },
    )
    expect(
        "windows search summary artifact names",
        search_contract.search_summary_artifact_names(runtime_kind="windows-cuda")
        == {
            "logs/metal-windows-cuda-mcts-bk07-search-summary.json",
            "logs/metal-windows-cuda-mcts-kiwipete-search-summary.json",
            "logs/metal-windows-cuda-mcts-after-e4-search-summary.json",
            "logs/metal-windows-cuda-hybrid-bk07-search-summary.json",
            "logs/metal-windows-cuda-hybrid-kiwipete-search-summary.json",
            "logs/metal-windows-cuda-hybrid-after-e4-search-summary.json",
        },
    )
    expect(
        "linux mcts bk07 path",
        linux["mcts_bk07"]
        == root / "metal-cuda-mcts-bk07-search-summary.json",
    )
    expect(
        "windows mcts bk07 path",
        windows["mcts_bk07"]
        == root / "logs" / "metal-windows-cuda-mcts-bk07-search-summary.json",
    )
    expect(
        "metal mcts bk07 artifact path",
        metal["mcts_bk07"] == root / "build" / "metal-mcts-bk07-search.json",
    )
    try:
        search_contract.search_summary_paths(root, runtime_kind="metal")
    except ValueError as exc:
        expect("unsupported search contract runtime", "unsupported" in str(exc))
        return
    raise AssertionError("expected unsupported runtime kind to be rejected")


def test_cuda_runtime_observed_parser_extracts_release_facts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        write_observed_runtime_inputs(root, runtime_kind="linux-cuda")
        observed = runtime_observed.collect_observed_runtime_facts(
            root,
            runtime_kind="linux-cuda",
        )
        benchmark = observed["benchmark_compare"]
        backend_after = benchmark["cuda_backend_after"]
        expect("observed benchmark present", benchmark["present"] is True)
        expect("observed graph replay", backend_after["executor_graph_replay"] is True)
        expect("observed replay count", backend_after["graph_replays"] == 58)
        expect(
            "observed stable batch",
            backend_after["cuda_stable_execution_batch_effective"] == 16,
        )
        expect(
            "observed stable batch ratio",
            abs(benchmark["stable_batch_eval_ms_ratio"] - 0.25) < 1e-9,
        )
        expect(
            "observed batch-one retained",
            benchmark["worst_eval_ms_ratio"] > 1.0,
        )
        expect(
            "observed search bestmove",
            observed["search_compare"]["mcts_bk07"]["cuda_bestmove"] == "h5f6",
        )


def runtime_manifest_writer_env(
    root: pathlib.Path,
    *,
    runtime_kind: str,
) -> tuple[pathlib.Path, dict[str, str]]:
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True)
    write_observed_runtime_inputs(artifact_dir, runtime_kind=runtime_kind)
    if runtime_kind == "linux-cuda":
        (artifact_dir / "cuda-gpu-tests.log").write_text("passed\n", encoding="utf-8")
    else:
        (artifact_dir / "logs" / "cuda-bk07-mcts.stdout.log").write_text(
            "bestmove h5f6\n", encoding="utf-8"
        )

    metal_dir = root / "metal"
    metal_dir.mkdir()

    def write_input(name: str) -> str:
        path = metal_dir / name
        path.write_text("ok\n", encoding="utf-8")
        return str(path)

    package = root / "metalfish-windows-x86_64-msvc-cuda.zip"
    package.write_bytes(b"package")
    archive = root / "metalfish.tar.gz"
    archive.write_bytes(b"archive")
    env = {
        "GIT_HEAD_SHA": "abc123",
        "GATE_ARTIFACT_DIR": str(artifact_dir),
        "GATE_PROJECT": "metalfish",
        "GATE_INSTANCE": "metalfish-cuda-test",
        "GATE_ZONE": "us-central1-a",
        "GATE_MACHINE": "g2-standard-8",
        "GATE_ACCELERATOR": "type=nvidia-l4,count=1",
        "GATE_IMAGE_PROJECT": "ubuntu-os-cloud",
        "GATE_IMAGE_FAMILY": "ubuntu-2204-lts",
        "GATE_BOOT_DISK_SIZE": "200GB",
        "GATE_DELETE_ON_EXIT": "1",
        "GATE_GCS_PREFIX": "",
        "GATE_REQUIRE_METAL_COMPARE": "1",
        "GATE_REQUIRE_METAL_BENCHMARK_COMPARE": "1",
        "GATE_REQUIRE_METAL_SEARCH_COMPARE": "1",
        "GATE_MAX_CUDA_METAL_EVAL_MS_RATIO": "1.0",
        "GATE_METAL_COMPARISON_LOG": write_input("metal-nn-comparison.log"),
        "GATE_METAL_PROBE_SUITE_LOG": write_input("metal-bt4.log"),
        "GATE_METAL_LEGACY_PROBE_SUITE_LOG": write_input("metal-legacy.log"),
        "GATE_METAL_MCTS_BK07_SEARCH_JSON": write_input("metal-mcts-bk07.json"),
        "GATE_METAL_MCTS_KIWIPETE_SEARCH_JSON": write_input(
            "metal-mcts-kiwipete.json"
        ),
        "GATE_METAL_MCTS_AFTER_E4_SEARCH_JSON": write_input(
            "metal-mcts-after-e4.json"
        ),
        "GATE_METAL_HYBRID_BK07_SEARCH_JSON": write_input(
            "metal-hybrid-bk07.json"
        ),
        "GATE_METAL_HYBRID_KIWIPETE_SEARCH_JSON": write_input(
            "metal-hybrid-kiwipete.json"
        ),
        "GATE_METAL_HYBRID_AFTER_E4_SEARCH_JSON": write_input(
            "metal-hybrid-after-e4.json"
        ),
        "GATE_CUDA_STABLE_BATCH_SIZE": "16",
        "GATE_CUDA_GRAPH": "1",
        "GATE_CUDA_GRAPH_EXECUTION": "1",
        "GATE_CUDA_DETERMINISTIC_ATTENTION_SOFTMAX": "1",
        "GATE_CUDA_FULL_BUFFER_CLEAR": "1",
        "GATE_CUDA_PROFILE": "0",
        "GATE_CUDA_PROFILE_LIMIT": "2",
        "GATE_CUBLAS_WORKSPACE_CONFIG": "",
        "BT4_COMPARE_STATUS_FOR_MANIFEST": "0",
        "LEGACY_COMPARE_STATUS_FOR_MANIFEST": "0",
        "BENCHMARK_COMPARE_STATUS_FOR_MANIFEST": "0",
        "SEARCH_COMPARE_STATUS_FOR_MANIFEST": "0",
        "FINAL_COMPARE_STATUS_FOR_MANIFEST": "0",
        "GATE_ARCHIVE": str(archive),
        "REMOTE_STATUS_FOR_MANIFEST": "0",
        "GATE_CUDA_UCI_GO": "nodes 8",
        "GATE_CUDA_MCTS_PONDER_GO": "wtime 60000 btime 60000",
        "GATE_CUDA_MCTS_PONDER_SETTLE_SEC": "0.6",
        "RUNTIME_STATUS_FOR_MANIFEST": "0",
        "GATE_MACHINES": "g2-standard-8 g2-standard-4",
        "GATE_BOOT_DISK_TYPE": "pd-balanced",
        "GATE_PACKAGE_ZIP": str(package),
        "GATE_PACKAGE_BASENAME": package.name,
        "GATE_WINDOWS_CUDA_COMPILE_RUN_ID": "12345",
        "GATE_UCI_GO": "nodes 8",
        "GATE_MCTS_TIMED_UCI_GO": "movetime 500",
        "GATE_MCTS_PONDER_UCI_GO": "wtime 60000 btime 60000",
        "GATE_MCTS_PONDER_SETTLE_MS": "600",
        "GATE_HYBRID_UCI_GO": "nodes 8",
        "GATE_HYBRID_PARITY_UCI_GO": "nodes 50",
    }
    return artifact_dir, env


def test_cuda_runtime_manifest_writer_keeps_linux_windows_schema_parity() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        for runtime_kind, manifest_name, expected_artifact in (
            ("linux-cuda", "cuda-gpu-runtime-manifest.json", "cuda-gpu-tests.log"),
            (
                "windows-cuda",
                "windows-cuda-runtime-gate-manifest.json",
                "logs/cuda-bk07-mcts.stdout.log",
            ),
        ):
            artifact_dir, env = runtime_manifest_writer_env(
                root / runtime_kind,
                runtime_kind=runtime_kind,
            )
            manifest = artifact_dir / manifest_name
            data = runtime_writer.write_manifest_from_env(
                runtime_kind=runtime_kind,
                manifest_path=manifest,
                env=env,
            )
            expect(
                f"{runtime_kind} artifact recorded",
                expected_artifact in data["artifacts"],
            )
            summary = runtime_checker.validate_runtime_manifest(
                manifest,
                runtime_kind=runtime_kind,
                require_metal_compare=True,
                require_metal_benchmark_compare=True,
                require_metal_search_compare=True,
                require_release_policy=True,
                require_observed_runtime=True,
                require_artifact_files=True,
                artifact_root=artifact_dir,
                expected_head_sha="abc123",
            )
            expect(f"{runtime_kind} schema", summary["kind"] == runtime_kind)
            expect(
                f"{runtime_kind} graph policy",
                summary["runtime"]["cuda_graph_execution"] == "1",
            )


def test_cuda_runtime_manifest_requires_timed_mcts_release_artifacts() -> None:
    linux_required = runtime_checker.REQUIRED_RELEASE_ARTIFACTS["linux-cuda"]
    windows_required = runtime_checker.REQUIRED_RELEASE_ARTIFACTS["windows-cuda"]
    for name in (
        "cuda-gpu-uci-timed-mcts-smoke.log",
        "cuda-gpu-uci-timed-mcts-search.json",
        "cuda-gpu-uci-after-e4-smoke.log",
        "cuda-gpu-uci-after-e4-search.json",
        "cuda-gpu-uci-hybrid-after-e4-smoke.log",
        "cuda-gpu-uci-hybrid-after-e4-search.json",
        "cuda-gpu-uci-hybrid-clock-start-smoke.log",
        "cuda-gpu-uci-hybrid-clock-safety-smoke.log",
    ):
        expect(f"linux release requires {name}", name in linux_required)
    for name in (
        "logs/cuda-timed-mcts.stdout.log",
        "logs/cuda-timed-mcts.stderr.log",
        "logs/cuda-timed-mcts-search.json",
        "logs/cuda-after-e4-mcts.stdout.log",
        "logs/cuda-after-e4-mcts-search.json",
        "logs/hybrid-cuda-after-e4.stdout.log",
        "logs/hybrid-cuda-after-e4-search.json",
        "logs/hybrid-cuda-clock-start.stdout.log",
        "logs/hybrid-cuda-clock-start.stderr.log",
        "logs/hybrid-cuda-clock-safety.stdout.log",
        "logs/hybrid-cuda-clock-safety.stderr.log",
    ):
        expect(f"windows release requires {name}", name in windows_required)


def test_cuda_release_artifact_helpers_validate_packages_and_manifests() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        linux_dir = root / "linux-package"
        windows_dir = root / "windows-package"
        write_release_package_files(linux_dir, package_kind="linux-cuda", windows=False)
        write_release_package_files(
            windows_dir, package_kind="windows-cuda", windows=True
        )
        linux_package = root / "metalfish-linux-x86_64-cuda.tar.gz"
        with tarfile.open(linux_package, "w:gz") as archive:
            for path in sorted(linux_dir.iterdir()):
                archive.add(path, arcname=path.name)
        windows_package = root / "metalfish-windows-x86_64-msvc-cuda.zip"
        with zipfile.ZipFile(windows_package, "w") as archive:
            for path in sorted(windows_dir.iterdir()):
                archive.write(path, arcname=path.name)

        linux_summary = cuda_release.validate_linux_cuda_package(
            linux_package,
            expected_source_commit="abc123",
        )
        windows_summary = cuda_release.validate_windows_cuda_package(
            windows_package,
            expected_source_commit="abc123",
        )
        expect("linux cuda kind", linux_summary["kind"] == "linux-cuda")
        expect("windows cuda kind", windows_summary["kind"] == "windows-cuda")
        expect("linux source commit", linux_summary["source_commit"] == "abc123")
        expect("windows source commit", windows_summary["source_commit"] == "abc123")

        linux_runtime = root / "cuda-gpu-runtime-manifest.json"
        linux_runtime.write_text(
            json.dumps(
                {
                    "schema": "metalfish.cuda_gpu_runtime_gate",
                    "git": {"head_sha": "abc123"},
                    "inputs": runtime_inputs(),
                    "status": {
                        "remote_status": "0",
                        "bt4_compare_status": "0",
                        "legacy_compare_status": "0",
                        "benchmark_compare_status": "0",
                        "search_compare_status": "0",
                        "final_compare_status": "0",
                    },
                    "artifacts": runtime_artifact_records(
                        "linux-cuda", linux_package=linux_package
                    ),
                }
            ),
            encoding="utf-8",
        )
        windows_runtime = root / "windows-cuda-runtime-gate-manifest.json"
        windows_runtime.write_text(
            json.dumps(
                {
                    "schema": "metalfish.windows_cuda_runtime_gate",
                    "git": {"head_sha": "abc123"},
                    "inputs": runtime_inputs(windows_package=windows_package),
                    "status": {
                        "runtime_status": "0",
                        "bt4_compare_status": "0",
                        "legacy_compare_status": "0",
                        "benchmark_compare_status": "0",
                        "search_compare_status": "0",
                        "final_compare_status": "0",
                    },
                    "artifacts": runtime_artifact_records("windows-cuda"),
                }
            ),
            encoding="utf-8",
        )
        expect(
            "linux runtime status",
            runtime_checker.validate_runtime_manifest(
                linux_runtime,
                runtime_kind="linux-cuda",
                require_metal_compare=True,
                require_metal_benchmark_compare=True,
                require_metal_search_compare=True,
                require_release_evidence=True,
                expected_head_sha="abc123",
            )["status"]["remote_status"]
            == "0",
        )
        expect(
            "windows runtime status",
            runtime_checker.validate_runtime_manifest(
                windows_runtime,
                runtime_kind="windows-cuda",
                require_metal_compare=True,
                require_metal_benchmark_compare=True,
                require_metal_search_compare=True,
                require_release_evidence=True,
                expected_head_sha="abc123",
            )["status"]["runtime_status"]
            == "0",
        )
        expect(
            "release package tag",
            cuda_release.release_package_name(
                linux_package, tag_name="v0.1.0-alpha", platform="linux-x86_64-cuda"
            )
            == "metalfish-v0.1.0-alpha-linux-x86_64-cuda.tar.gz",
        )


def test_cuda_package_validator_rejects_unmanifested_archive_entries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        windows_dir = root / "windows-package"
        write_release_package_files(
            windows_dir, package_kind="windows-cuda", windows=True
        )
        extra = windows_dir / "unexpected.dll"
        extra.write_text("unmanifested\n", encoding="utf-8")
        windows_package = root / "metalfish-windows-x86_64-msvc-cuda.zip"
        with zipfile.ZipFile(windows_package, "w") as archive:
            for path in sorted(windows_dir.iterdir()):
                archive.write(path, arcname=path.name)
        try:
            cuda_release.validate_windows_cuda_package(
                windows_package,
                expected_source_commit="abc123",
            )
        except ValueError as exc:
            expect("unmanifested entry rejected", "unmanifested entries" in str(exc))
            return
    raise AssertionError("expected unmanifested Windows CUDA archive entry rejection")


def test_cuda_release_artifacts_promote_direct_runtime_root() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        direct = root / "direct"
        linux_package_dir = root / "linux-package"
        windows_package_dir = root / "windows-package"
        write_release_package_files(
            linux_package_dir,
            package_kind="linux-cuda",
            windows=False,
            source_commit="abc123",
        )
        write_release_package_files(
            windows_package_dir,
            package_kind="windows-cuda",
            windows=True,
            source_commit="abc123",
        )
        linux_dir = direct / "linux"
        windows_dir = direct / "windows"
        windows_inputs_dir = direct / "windows_cuda_runtime_inputs" / "windows"
        linux_dir.mkdir(parents=True)
        windows_dir.mkdir(parents=True)
        windows_inputs_dir.mkdir(parents=True)
        linux_package = linux_dir / "metalfish-linux-x86_64-cuda.tar.gz"
        with tarfile.open(linux_package, "w:gz") as archive:
            for path in sorted(linux_package_dir.iterdir()):
                archive.add(path, arcname=path.name)
        windows_package = windows_inputs_dir / "metalfish-windows-x86_64-msvc-cuda.zip"
        with zipfile.ZipFile(windows_package, "w") as archive:
            for path in sorted(windows_package_dir.iterdir()):
                archive.write(path, arcname=path.name)

        (direct / "direct-runtime-gates-manifest.json").write_text(
            json.dumps(
                {
                    "schema": "metalfish.cuda_runtime_gates_direct",
                    "expected_sha": "abc123",
                    "repo": "owner/repo",
                    "target": "both",
                    "require_metal": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (linux_dir / "cuda-gpu-runtime-manifest.json").write_text(
            json.dumps(
                {
                    "schema": "metalfish.cuda_gpu_runtime_gate",
                    "git": {"head_sha": "abc123"},
                    "inputs": runtime_inputs(),
                    "status": {
                        "remote_status": "0",
                        "bt4_compare_status": "0",
                        "legacy_compare_status": "0",
                        "benchmark_compare_status": "0",
                        "search_compare_status": "0",
                        "final_compare_status": "0",
                    },
                    "runtime": runtime_policy(),
                    "observed_runtime": observed_runtime_facts(
                        runtime_kind="linux-cuda"
                    ),
                    "artifacts": runtime_artifact_records(
                        "linux-cuda",
                        linux_package=linux_package,
                        artifact_root=linux_dir,
                    ),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (windows_dir / "windows-cuda-runtime-gate-manifest.json").write_text(
            json.dumps(
                {
                    "schema": "metalfish.windows_cuda_runtime_gate",
                    "git": {"head_sha": "abc123"},
                    "inputs": runtime_inputs(windows_package=windows_package),
                    "status": {
                        "runtime_status": "0",
                        "bt4_compare_status": "0",
                        "legacy_compare_status": "0",
                        "benchmark_compare_status": "0",
                        "search_compare_status": "0",
                        "final_compare_status": "0",
                    },
                    "runtime": runtime_policy(),
                    "observed_runtime": observed_runtime_facts(
                        runtime_kind="windows-cuda"
                    ),
                    "artifacts": runtime_artifact_records(
                        "windows-cuda",
                        artifact_root=windows_dir,
                    ),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        out_dir = root / "release"
        expect(
            "direct release promotion",
            cuda_release.main(
                [
                    "--direct-runtime-root",
                    str(direct),
                    "--out-dir",
                    str(out_dir),
                    "--tag-name",
                    "v0.1.0-alpha",
                ]
            )
            == 0,
        )
        manifest = json.loads(
            (out_dir / "cuda-release-artifacts-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        expect("direct repo", manifest["repo"] == "owner/repo")
        expect("direct expected sha", manifest["expected_sha"] == "abc123")
        expect(
            "direct linux mode",
            manifest["runs"]["linux_cuda"]["mode"] == "direct-runtime-root",
        )
        expect(
            "direct windows source",
            manifest["packages"]["windows_cuda"]["manifest"]["source_commit"]
            == "abc123",
        )
        expect(
            "direct release package",
            (
                out_dir
                / "packages"
                / "metalfish-v0.1.0-alpha-windows-x86_64-msvc-cuda.zip"
            ).is_file(),
        )


def test_cuda_release_artifacts_rejects_direct_output_inside_input() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        direct = root / "direct"
        direct.mkdir()
        (direct / "direct-runtime-gates-manifest.json").write_text(
            json.dumps(
                {
                    "schema": "metalfish.cuda_runtime_gates_direct",
                    "expected_sha": "abc123",
                    "repo": "owner/repo",
                    "target": "both",
                    "require_metal": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        for out_dir in (direct, root):
            try:
                cuda_release.main(
                    [
                        "--direct-runtime-root",
                        str(direct),
                        "--out-dir",
                        str(out_dir),
                        "--expected-sha",
                        "abc123",
                    ]
                )
            except ValueError as exc:
                expect("direct output overlap rejected", "--out-dir" in str(exc))
            else:
                raise AssertionError("expected direct runtime output overlap rejection")


def test_cuda_release_artifacts_reject_direct_runtime_single_target() -> None:
    try:
        cuda_release.validate_direct_runtime_manifest(
            {
                "schema": "metalfish.cuda_runtime_gates_direct",
                "expected_sha": "abc123",
                "repo": "owner/repo",
                "target": "windows",
                "require_metal": True,
            },
            expected_sha="abc123",
        )
    except ValueError as exc:
        expect("direct single-target failure", "both Linux and Windows" in str(exc))
        return
    raise AssertionError("expected single-target direct runtime manifest to fail")


def test_cuda_release_artifacts_reject_direct_runtime_commit_drift() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        direct = root / "direct"
        linux_package_dir = root / "linux-package"
        windows_package_dir = root / "windows-package"
        write_release_package_files(
            linux_package_dir,
            package_kind="linux-cuda",
            windows=False,
            source_commit="linux-sha",
        )
        write_release_package_files(
            windows_package_dir,
            package_kind="windows-cuda",
            windows=True,
            source_commit="windows-sha",
        )
        linux_dir = direct / "linux"
        windows_dir = direct / "windows"
        windows_inputs_dir = direct / "windows_cuda_runtime_inputs" / "windows"
        linux_dir.mkdir(parents=True)
        windows_dir.mkdir(parents=True)
        windows_inputs_dir.mkdir(parents=True)
        linux_package = linux_dir / "metalfish-linux-x86_64-cuda.tar.gz"
        with tarfile.open(linux_package, "w:gz") as archive:
            for path in sorted(linux_package_dir.iterdir()):
                archive.add(path, arcname=path.name)
        windows_package = windows_inputs_dir / "metalfish-windows-x86_64-msvc-cuda.zip"
        with zipfile.ZipFile(windows_package, "w") as archive:
            for path in sorted(windows_package_dir.iterdir()):
                archive.write(path, arcname=path.name)

        (direct / "direct-runtime-gates-manifest.json").write_text(
            json.dumps(
                {
                    "schema": "metalfish.cuda_runtime_gates_direct",
                    "expected_sha": "linux-sha",
                    "repo": "owner/repo",
                    "target": "both",
                    "require_metal": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (linux_dir / "cuda-gpu-runtime-manifest.json").write_text(
            json.dumps(
                {
                    "schema": "metalfish.cuda_gpu_runtime_gate",
                    "git": {"head_sha": "linux-sha"},
                    "inputs": runtime_inputs(),
                    "status": {
                        "remote_status": "0",
                        "bt4_compare_status": "0",
                        "legacy_compare_status": "0",
                        "benchmark_compare_status": "0",
                        "search_compare_status": "0",
                        "final_compare_status": "0",
                    },
                    "artifacts": runtime_artifact_records(
                        "linux-cuda", linux_package=linux_package
                    ),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (windows_dir / "windows-cuda-runtime-gate-manifest.json").write_text(
            json.dumps(
                {
                    "schema": "metalfish.windows_cuda_runtime_gate",
                    "git": {"head_sha": "windows-sha"},
                    "inputs": runtime_inputs(windows_package=windows_package),
                    "status": {
                        "runtime_status": "0",
                        "bt4_compare_status": "0",
                        "legacy_compare_status": "0",
                        "benchmark_compare_status": "0",
                        "search_compare_status": "0",
                        "final_compare_status": "0",
                    },
                    "artifacts": runtime_artifact_records("windows-cuda"),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        try:
            cuda_release.main(
                [
                    "--direct-runtime-root",
                    str(direct),
                    "--out-dir",
                    str(root / "release"),
                    "--tag-name",
                    "v0.1.0-alpha",
                ]
            )
        except ValueError as exc:
            expect("direct commit drift rejected", "source commit" in str(exc))
            return
    raise AssertionError("expected direct runtime commit drift to be rejected")


def test_cuda_release_dispatch_direct_root_uses_manifest_sha_default() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        direct_root = root / "direct"
        out_dir = root / "release"
        direct_root.mkdir()
        calls: list[list[str]] = []
        old_main = cuda_release_dispatch.cuda_release.main

        def fake_promote(args: list[str]) -> int:
            calls.append(list(args))
            return 0

        cuda_release_dispatch.cuda_release.main = fake_promote
        try:
            expect(
                "direct dispatch promotion",
                cuda_release_dispatch.main(
                    [
                        "--direct-runtime-root",
                        str(direct_root),
                        "--out-dir",
                        str(out_dir),
                        "--tag-name",
                        "v0.1.0-alpha",
                    ]
                )
                == 0,
            )
        finally:
            cuda_release_dispatch.cuda_release.main = old_main

        expect("direct dispatch called promote", len(calls) == 1)
        expect("direct root forwarded", "--direct-runtime-root" in calls[0])
        expect("out dir forwarded", "--out-dir" in calls[0])
        expect("tag forwarded", "--tag-name" in calls[0])
        expect(
            "direct dispatch does not default to checkout head",
            "--expected-sha" not in calls[0],
        )


def test_cuda_release_dispatch_validates_explicit_run_ids() -> None:
    calls: list[list[str]] = []
    old_run_json = cuda_release_dispatch.dispatch.run_json

    def fake_run_json(cmd: list[str]) -> dict:
        calls.append(list(cmd))
        return {
            "conclusion": "success",
            "createdAt": "2026-05-29T00:00:00Z",
            "databaseId": 12345,
            "headSha": "abc123",
            "status": "completed",
            "url": "https://example.invalid/run/12345",
            "workflowName": "CUDA GPU Gate",
        }

    cuda_release_dispatch.dispatch.run_json = fake_run_json
    try:
        run = cuda_release_dispatch.run_successful_gate(
            repo="owner/repo",
            workflow="CUDA GPU Gate",
            gate_ref="cuda-support",
            expected_sha="abc123",
            run_id="12345",
            lookup_limit=30,
        )
    finally:
        cuda_release_dispatch.dispatch.run_json = old_run_json

    expect("explicit run validated", run["databaseId"] == 12345)
    expect(
        "explicit run uses gh view",
        calls and calls[0][0:3] == ["gh", "run", "view"],
    )


def test_cuda_release_artifacts_accept_split_direct_runtime_manifest() -> None:
    cuda_release.validate_direct_runtime_manifest(
        {
            "schema": "metalfish.cuda_runtime_gates_direct",
            "expected_sha": "abc123",
            "repo": "owner/repo",
            "target": "windows",
            "completed_targets": ["linux", "windows"],
            "require_metal": True,
        },
        expected_sha="abc123",
    )


def test_direct_runtime_manifest_merge_preserves_split_targets() -> None:
    existing = {
        "schema": "metalfish.cuda_runtime_gates_direct",
        "schema_version": 1,
        "created_at_unix": 100,
        "repo": "owner/repo",
        "ref": "cuda-support",
        "expected_sha": "abc123",
        "target": "linux",
        "require_metal": True,
        "metal_ci_run_id": "1",
        "windows_cuda_run_id": None,
        "linux_instance": "linux-vm",
        "linux_machine": "g2-standard-8",
        "windows_instance": None,
        "windows_machines": None,
    }
    current = {
        "schema": "metalfish.cuda_runtime_gates_direct",
        "schema_version": 1,
        "created_at_unix": 200,
        "repo": "owner/repo",
        "ref": "cuda-support",
        "expected_sha": "abc123",
        "target": "windows",
        "require_metal": True,
        "metal_ci_run_id": "1",
        "windows_cuda_run_id": "2",
        "linux_instance": None,
        "linux_machine": None,
        "windows_instance": "windows-vm",
        "windows_machines": "g2-standard-8 g2-standard-4",
    }

    merged = direct_runtime.merge_direct_runtime_manifest(existing, current)
    expect("merged direct target", merged["target"] == "both")
    expect(
        "merged completed targets",
        merged["completed_targets"] == ["linux", "windows"],
    )
    expect("merged linux instance", merged["linux_instance"] == "linux-vm")
    expect("merged windows instance", merged["windows_instance"] == "windows-vm")
    expect("merged windows run id", merged["windows_cuda_run_id"] == "2")
    expect("merged created preserved", merged["created_at_unix"] == 100)
    expect("merged updated recorded", merged["updated_at_unix"] == 200)


def test_direct_runtime_clears_only_selected_artifact_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        artifact_root = root / "artifacts"
        linux_dir = artifact_root / "linux"
        windows_dir = artifact_root / "windows"
        linux_dir.mkdir(parents=True)
        windows_dir.mkdir(parents=True)
        (linux_dir / "stale-package.tar.gz").write_text("old", encoding="utf-8")
        (windows_dir / "kept-package.zip").write_text("old", encoding="utf-8")

        calls: list[tuple[list[str], pathlib.Path, dict[str, str] | None]] = []
        old_resolve_runs = direct_runtime.resolve_runs
        old_run_command = direct_runtime.run_command

        def fake_resolve_runs(
            args: object, repo: str, ref: str, expected_sha: str
        ) -> dict:
            return {
                "metal_ci_run_id": "",
                "windows_cuda_run_id": "",
                "metal_run": None,
                "windows_run": None,
            }

        def fake_run_command(
            cmd: list[str],
            *,
            cwd: pathlib.Path,
            env: dict[str, str] | None = None,
            dry_run: bool,
        ) -> None:
            calls.append((list(cmd), cwd, None if env is None else dict(env)))

        direct_runtime.resolve_runs = fake_resolve_runs
        direct_runtime.run_command = fake_run_command
        try:
            expect(
                "direct runtime selected cleanup",
                direct_runtime.main(
                    [
                        "--repo",
                        "owner/repo",
                        "--ref",
                        "cuda-support",
                        "--expected-sha",
                        "abc123",
                        "--target",
                        "linux",
                        "--no-require-metal",
                        "--no-cuda-graph-execution",
                        "--profile",
                        "--profile-limit",
                        "3",
                        "--cublas-workspace-config",
                        ":4096:8",
                        "--artifact-root",
                        str(artifact_root),
                        "--worktree-dir",
                        str(root / "worktree"),
                    ]
                )
                == 0,
            )
        finally:
            direct_runtime.resolve_runs = old_resolve_runs
            direct_runtime.run_command = old_run_command

        expect("selected linux artifacts cleared", not linux_dir.exists())
        expect(
            "non-selected windows artifacts kept",
            (windows_dir / "kept-package.zip").is_file(),
        )
        expect(
            "linux gate still targeted selected dir",
            any(
                env is not None
                and env.get("METALFISH_GCP_ARTIFACT_DIR") == str(linux_dir.resolve())
                for _, _, env in calls
            ),
        )
        expect(
            "direct runtime graph disable propagated",
            any(
                env is not None and env.get("METALFISH_CUDA_GRAPH_EXECUTION") == "0"
                for _, _, env in calls
            ),
        )
        expect(
            "direct runtime profile propagated",
            any(
                env is not None
                and env.get("METALFISH_CUDA_PROFILE") == "1"
                and env.get("METALFISH_CUDA_PROFILE_LIMIT") == "3"
                for _, _, env in calls
            ),
        )
        expect(
            "direct runtime cublas config propagated",
            any(
                env is not None and env.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
                for _, _, env in calls
            ),
        )


def test_cuda_release_artifact_helpers_reject_failed_runtime() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest = pathlib.Path(tmp) / "windows-cuda-runtime-gate-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "metalfish.windows_cuda_runtime_gate",
                    "status": {
                        "runtime_status": "0",
                        "bt4_compare_status": "1",
                        "legacy_compare_status": "0",
                        "final_compare_status": "1",
                    },
                }
            ),
            encoding="utf-8",
        )
        try:
            runtime_checker.validate_runtime_manifest(
                manifest, runtime_kind="windows-cuda"
            )
        except ValueError as exc:
            expect("bt4 compare failure", "BT4 compare status" in str(exc))
            return
    raise AssertionError("expected failed runtime manifest to be rejected")


def test_cuda_runtime_manifest_rejects_head_sha_drift() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest = pathlib.Path(tmp) / "runtime.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "metalfish.cuda_gpu_runtime_gate",
                    "git": {"head_sha": "old-sha"},
                    "status": {
                        "remote_status": "0",
                        "bt4_compare_status": "0",
                        "legacy_compare_status": "0",
                        "final_compare_status": "0",
                    },
                }
            ),
            encoding="utf-8",
        )
        try:
            runtime_checker.validate_runtime_manifest(
                manifest,
                runtime_kind="linux-cuda",
                expected_head_sha="new-sha",
            )
        except ValueError as exc:
            expect("manifest head drift rejected", "git head" in str(exc))
            return
    raise AssertionError("expected runtime manifest head drift to be rejected")


def test_cuda_runtime_manifest_validates_artifact_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        artifact_root = root / "artifacts"
        payload = artifact_root / "logs" / "cuda-bk07-mcts-search.json"
        payload.parent.mkdir(parents=True)
        payload.write_text('{"bestmove":"h5f6"}\n', encoding="utf-8")
        record = cuda_release.file_record(payload)
        record["path"] = "original-runner-path/cuda-bk07-mcts-search.json"
        manifest = root / "runtime.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "metalfish.windows_cuda_runtime_gate",
                    "git": {"head_sha": "abc123"},
                    "status": {
                        "runtime_status": "0",
                        "bt4_compare_status": "0",
                        "legacy_compare_status": "0",
                        "final_compare_status": "0",
                    },
                    "artifacts": {"logs/cuda-bk07-mcts-search.json": record},
                }
            ),
            encoding="utf-8",
        )
        summary = runtime_checker.validate_runtime_manifest(
            manifest,
            runtime_kind="windows-cuda",
            require_artifact_files=True,
            artifact_root=artifact_root,
            expected_head_sha="abc123",
        )
        expect(
            "artifact file validation summary",
            "logs/cuda-bk07-mcts-search.json" in summary["artifacts"],
        )
        payload.write_text('{"bestmove":"a1a1"}\n', encoding="utf-8")
        try:
            runtime_checker.validate_runtime_manifest(
                manifest,
                runtime_kind="windows-cuda",
                require_artifact_files=True,
                artifact_root=artifact_root,
            )
        except ValueError as exc:
            expect("artifact hash mismatch rejected", "sha256 mismatch" in str(exc))
            return
    raise AssertionError("expected artifact hash drift to be rejected")


def test_cuda_runtime_manifest_rejects_artifacts_outside_root() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        outside = root / "outside" / "cuda-bk07-mcts-search.json"
        outside.parent.mkdir(parents=True)
        outside.write_text('{"bestmove":"h5f6"}\n', encoding="utf-8")
        record = cuda_release.file_record(outside)
        manifest = root / "runtime.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "metalfish.windows_cuda_runtime_gate",
                    "git": {"head_sha": "abc123"},
                    "status": {
                        "runtime_status": "0",
                        "bt4_compare_status": "0",
                        "legacy_compare_status": "0",
                        "final_compare_status": "0",
                    },
                    "artifacts": {"logs/cuda-bk07-mcts-search.json": record},
                }
            ),
            encoding="utf-8",
        )
        try:
            runtime_checker.validate_runtime_manifest(
                manifest,
                runtime_kind="windows-cuda",
                require_artifact_files=True,
                artifact_root=root / "artifacts",
            )
        except ValueError as exc:
            expect("outside artifact rejected", "file is missing" in str(exc))
            return
    raise AssertionError("expected outside-root artifact path to be rejected")


def test_cuda_runtime_manifest_enforces_release_policy() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manifest = pathlib.Path(tmp) / "runtime.json"
        base = {
            "schema": "metalfish.windows_cuda_runtime_gate",
            "git": {"head_sha": "abc123"},
            "inputs": runtime_inputs(),
            "status": {
                "runtime_status": "0",
                "bt4_compare_status": "0",
                "legacy_compare_status": "0",
                "final_compare_status": "0",
            },
        }
        good = dict(base)
        good["runtime"] = runtime_policy()
        good["observed_runtime"] = observed_runtime_facts()
        manifest.write_text(json.dumps(good), encoding="utf-8")
        summary = runtime_checker.validate_runtime_manifest(
            manifest,
            runtime_kind="windows-cuda",
            require_release_policy=True,
            require_observed_runtime=True,
        )
        expect(
            "release runtime policy accepted",
            summary["runtime"]["cuda_stable_execution_batch_size"] == "16",
        )

        def expect_policy_rejection(policy: dict, text: str) -> None:
            bad = dict(base)
            bad["runtime"] = policy
            manifest.write_text(json.dumps(bad), encoding="utf-8")
            try:
                runtime_checker.validate_runtime_manifest(
                    manifest,
                    runtime_kind="windows-cuda",
                    require_release_policy=True,
                    require_observed_runtime=True,
                )
            except ValueError as exc:
                expect(f"release policy rejected {text}", text in str(exc))
                return
            raise AssertionError(f"expected release runtime policy rejection: {text}")

        expect_policy_rejection(runtime_policy(profile="1"), "profiling disabled")
        expect_policy_rejection(runtime_policy(graph=""), "graph enabled")
        expect_policy_rejection(
            runtime_policy(deterministic_attention_softmax="0"),
            "cuda_deterministic_attention_softmax",
        )
        expect_policy_rejection(
            runtime_policy(full_buffer_clear="0"),
            "cuda_full_buffer_clear",
        )
        expect_policy_rejection(
            runtime_policy(cublas_workspace=":4096:8"),
            "CUBLAS_WORKSPACE_CONFIG",
        )

        bad = dict(base)
        bad["inputs"] = dict(runtime_inputs())
        bad["inputs"]["max_cuda_metal_eval_ms_ratio"] = "1.25"
        bad["runtime"] = runtime_policy()
        bad["observed_runtime"] = observed_runtime_facts()
        manifest.write_text(json.dumps(bad), encoding="utf-8")
        try:
            runtime_checker.validate_runtime_manifest(
                manifest,
                runtime_kind="windows-cuda",
                require_release_policy=True,
                require_observed_runtime=True,
            )
        except ValueError as exc:
            expect("relaxed ratio rejected", "Metal comparison ratio" in str(exc))
        else:
            raise AssertionError("expected relaxed Metal comparison ratio rejection")

        for observed, text in (
            (observed_runtime_facts(graph_replay=False), "cuda_graph_effective"),
            (observed_runtime_facts(stable_batch=8), "stable batch"),
            (
                observed_runtime_facts(deterministic_attention_softmax=False),
                "deterministic attention softmax",
            ),
            (observed_runtime_facts(full_buffer_clear=False), "full buffer clear"),
            (
                observed_runtime_facts(stable_batch_eval_ms_ratio=1.01),
                "eval-ms ratio",
            ),
            (observed_runtime_facts(search_status="failed"), "search did not pass"),
            (
                observed_runtime_facts(same_bestmove_required=False),
                "same bestmove",
            ),
            (observed_runtime_facts(bestmove_matches=False), "bestmove mismatch"),
        ):
            bad = dict(base)
            bad["runtime"] = runtime_policy()
            bad["observed_runtime"] = observed
            manifest.write_text(json.dumps(bad), encoding="utf-8")
            try:
                runtime_checker.validate_runtime_manifest(
                    manifest,
                    runtime_kind="windows-cuda",
                    require_release_policy=True,
                    require_observed_runtime=True,
                )
            except ValueError as exc:
                expect(f"observed runtime rejected {text}", text in str(exc))
            else:
                raise AssertionError(f"expected observed runtime rejection: {text}")


def test_cuda_package_validator_rejects_source_commit_drift() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        package_dir = root / "linux-package"
        write_release_package_files(
            package_dir,
            package_kind="linux-cuda",
            windows=False,
            source_commit="old-sha",
        )
        package = root / "metalfish-linux-x86_64-cuda.tar.gz"
        with tarfile.open(package, "w:gz") as archive:
            for path in sorted(package_dir.iterdir()):
                archive.add(path, arcname=path.name)
        try:
            cuda_release.validate_linux_cuda_package(
                package,
                expected_source_commit="new-sha",
            )
        except ValueError as exc:
            expect("source commit drift rejected", "source commit" in str(exc))
            return
    raise AssertionError("expected source commit drift to be rejected")


def test_cuda_package_validator_rejects_hash_drift() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        package_dir = root / "linux-package"
        write_release_package_files(
            package_dir,
            package_kind="linux-cuda",
            windows=False,
            source_commit="abc123",
        )
        (package_dir / "metalfish").write_text("tampered!\n", encoding="utf-8")
        package = root / "metalfish-linux-x86_64-cuda.tar.gz"
        with tarfile.open(package, "w:gz") as archive:
            for path in sorted(package_dir.iterdir()):
                archive.add(path, arcname=path.name)
        try:
            cuda_release.validate_linux_cuda_package(
                package,
                expected_source_commit="abc123",
            )
        except ValueError as exc:
            expect("package hash drift rejected", "sha256 mismatch" in str(exc))
            return
    raise AssertionError("expected package hash drift to be rejected")


def test_cuda_release_artifact_helpers_require_metal_compare() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        missing = root / "missing-metal.json"
        missing.write_text(
            json.dumps(
                {
                    "schema": "metalfish.cuda_gpu_runtime_gate",
                    "inputs": {"require_metal_compare": "0"},
                    "status": {
                        "remote_status": "0",
                        "bt4_compare_status": "0",
                        "legacy_compare_status": "0",
                        "final_compare_status": "0",
                    },
                }
            ),
            encoding="utf-8",
        )
        try:
            runtime_checker.validate_runtime_manifest(
                missing,
                runtime_kind="linux-cuda",
                require_metal_compare=True,
            )
        except ValueError as exc:
            expect("missing metal rejected", "require Metal comparison" in str(exc))
        else:
            raise AssertionError("expected missing Metal comparison to be rejected")

        incomplete = root / "incomplete-metal.json"
        incomplete.write_text(
            json.dumps(
                {
                    "schema": "metalfish.windows_cuda_runtime_gate",
                    "inputs": {
                        "require_metal_compare": "1",
                        "metal_probe_suite_log": metal_log_record("metal-bt4.log"),
                    },
                    "status": {
                        "runtime_status": "0",
                        "bt4_compare_status": "0",
                        "legacy_compare_status": "0",
                        "final_compare_status": "0",
                    },
                }
            ),
            encoding="utf-8",
        )
        try:
            runtime_checker.validate_runtime_manifest(
                incomplete,
                runtime_kind="windows-cuda",
                require_metal_compare=True,
            )
        except ValueError as exc:
            expect("legacy metal rejected", "Metal legacy probe suite" in str(exc))
            return
    raise AssertionError("expected incomplete Metal comparison to be rejected")


def test_cuda_runtime_manifest_requires_benchmark_compare() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        missing_status = root / "missing-benchmark-status.json"
        missing_status.write_text(
            json.dumps(
                {
                    "schema": "metalfish.cuda_gpu_runtime_gate",
                    "inputs": runtime_inputs(),
                    "status": {
                        "remote_status": "0",
                        "bt4_compare_status": "0",
                        "legacy_compare_status": "0",
                        "final_compare_status": "0",
                    },
                }
            ),
            encoding="utf-8",
        )
        try:
            runtime_checker.validate_runtime_manifest(
                missing_status,
                runtime_kind="linux-cuda",
                require_metal_compare=True,
                require_metal_benchmark_compare=True,
            )
        except ValueError as exc:
            expect("benchmark status rejected", "benchmark compare status" in str(exc))
        else:
            raise AssertionError("expected missing benchmark status to be rejected")

        missing_input = root / "missing-benchmark-input.json"
        missing_input.write_text(
            json.dumps(
                {
                    "schema": "metalfish.windows_cuda_runtime_gate",
                    "inputs": {
                        "require_metal_compare": "1",
                        "require_metal_benchmark_compare": "0",
                        "metal_probe_suite_log": metal_log_record("metal-bt4.log"),
                        "metal_legacy_probe_suite_log": metal_log_record(
                            "metal-legacy.log"
                        ),
                    },
                    "status": {
                        "runtime_status": "0",
                        "bt4_compare_status": "0",
                        "legacy_compare_status": "0",
                        "benchmark_compare_status": "0",
                        "final_compare_status": "0",
                    },
                }
            ),
            encoding="utf-8",
        )
        try:
            runtime_checker.validate_runtime_manifest(
                missing_input,
                runtime_kind="windows-cuda",
                require_metal_compare=True,
                require_metal_benchmark_compare=True,
            )
        except ValueError as exc:
            expect("benchmark input rejected", "benchmark comparison" in str(exc))
            return
    raise AssertionError("expected missing benchmark input to be rejected")


def test_network_downloader_validates_gzip_weights() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        valid = root / "valid.pb.gz"
        invalid = root / "invalid.pb.gz"
        with gzip.open(valid, "wb") as handle:
            handle.write(downloader.WEIGHTS_PROTO_MAGIC_PREFIX + b"synthetic")
        with gzip.open(invalid, "wb") as handle:
            handle.write(b"not-a-weights-protobuf")

        downloader.validate_gzip_weights(valid)
        try:
            downloader.validate_gzip_weights(invalid)
        except RuntimeError as exc:
            expect("invalid magic rejected", "Lc0 protobuf" in str(exc))
            return
    raise AssertionError("expected invalid gzip weights to be rejected")


def test_network_downloader_retries_truncated_gzip() -> None:
    valid_payload = gzip.compress(downloader.WEIGHTS_PROTO_MAGIC_PREFIX + b"synthetic")
    truncated_payload = valid_payload[:-8]
    calls = 0

    class Response(io.BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    old_urlopen = downloader.urllib.request.urlopen

    def fake_urlopen(_request: object, timeout: int) -> Response:
        nonlocal calls
        expect("download timeout forwarded", timeout == 60)
        calls += 1
        return Response(truncated_payload if calls == 1 else valid_payload)

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        dest = root / "weights.pb.gz"
        downloader.urllib.request.urlopen = fake_urlopen
        try:
            downloader.download(
                "https://example.invalid/weights.pb.gz",
                dest,
                retries=2,
                force=False,
                validator=downloader.validate_gzip_weights,
            )
        finally:
            downloader.urllib.request.urlopen = old_urlopen

        expect("retried after truncated gzip", calls == 2)
        expect("downloaded destination exists", dest.exists())
        expect("temporary file cleaned", not dest.with_suffix(dest.suffix + ".tmp").exists())
        downloader.validate_gzip_weights(dest)


def test_network_downloader_uses_validated_cache_before_network() -> None:
    valid_payload = gzip.compress(downloader.WEIGHTS_PROTO_MAGIC_PREFIX + b"synthetic")

    class Response(io.BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    old_urlopen = downloader.urllib.request.urlopen

    def fake_urlopen(_request: object, timeout: int) -> Response:
        raise AssertionError("cache hit should not open network URL")

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        cache = root / "cache"
        output = root / "output"
        cache.mkdir()
        output.mkdir()
        source = cache / "weights.pb.gz"
        dest = output / "weights.pb.gz"
        source.write_bytes(valid_payload)

        downloader.urllib.request.urlopen = fake_urlopen
        try:
            downloader.download(
                "https://example.invalid/weights.pb.gz",
                dest,
                retries=1,
                force=False,
                validator=downloader.validate_gzip_weights,
                cache_dirs=[cache],
            )
        finally:
            downloader.urllib.request.urlopen = old_urlopen

        expect("cached network copied", dest.read_bytes() == valid_payload)
        downloader.validate_gzip_weights(dest)


def test_network_downloader_writes_valid_download_to_cache() -> None:
    valid_payload = gzip.compress(downloader.WEIGHTS_PROTO_MAGIC_PREFIX + b"synthetic")
    calls = 0

    class Response(io.BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    old_urlopen = downloader.urllib.request.urlopen

    def fake_urlopen(_request: object, timeout: int) -> Response:
        nonlocal calls
        expect("download timeout forwarded", timeout == 60)
        calls += 1
        return Response(valid_payload)

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        cache = root / "cache"
        output = root / "output"
        output.mkdir()
        dest = output / "weights.pb.gz"

        downloader.urllib.request.urlopen = fake_urlopen
        try:
            downloader.download(
                "https://example.invalid/weights.pb.gz",
                dest,
                retries=1,
                force=False,
                validator=downloader.validate_gzip_weights,
                cache_dirs=[cache],
            )
        finally:
            downloader.urllib.request.urlopen = old_urlopen

        expect("downloaded exactly once", calls == 1)
        expect("downloaded destination exists", dest.read_bytes() == valid_payload)
        expect(
            "network cache populated",
            (cache / "weights.pb.gz").read_bytes() == valid_payload,
        )
        downloader.validate_gzip_weights(cache / "weights.pb.gz")


def test_gcp_cuda_resource_audit_filters_instances() -> None:
    payload = json.dumps(
        [
            {
                "name": "metalfish-cuda-direct-linux-old",
                "zone": (
                    "https://www.googleapis.com/compute/v1/projects/p/"
                    "zones/us-central1-a"
                ),
                "status": "RUNNING",
                "creationTimestamp": "2026-05-29T02:00:00.000-00:00",
            },
            {
                "name": "metalfish-win-cuda-gh-new",
                "zone": "us-central1-b",
                "status": "TERMINATED",
                "creationTimestamp": "2026-05-29T06:45:00Z",
            },
            {
                "name": "unrelated",
                "zone": "us-central1-c",
                "status": "RUNNING",
                "creationTimestamp": "2026-05-29T01:00:00Z",
            },
        ]
    )
    instances = gcp_audit.parse_instances(payload)
    matches = gcp_audit.filter_instances(
        instances,
        name_regex=gcp_audit.DEFAULT_NAME_REGEX,
        older_than_hours=2,
        now=datetime(2026, 5, 29, 7, 0, tzinfo=timezone.utc),
    )

    expect(
        "old CUDA instance selected",
        [instance.name for instance in matches] == ["metalfish-cuda-direct-linux-old"],
    )
    expect("zone URL normalized", matches[0].zone == "us-central1-a")


def test_portable_manifest_uses_checked_out_commit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        binary = root / "metalfish"
        output = root / "PORTABLE_ARTIFACT.md"
        manifest = root / "manifest.json"
        binary.write_text("synthetic", encoding="utf-8")

        old_env = {
            key: os.environ.get(key)
            for key in ("GITHUB_SHA", "GITHUB_HEAD_REF", "GITHUB_REF_NAME")
        }
        os.environ["GITHUB_SHA"] = "merge-commit"
        os.environ["GITHUB_HEAD_REF"] = "cuda-support"
        os.environ["GITHUB_REF_NAME"] = "43/merge"
        try:
            with argv(
                [
                    "--platform",
                    "Synthetic",
                    "--backend",
                    "Synthetic backend",
                    "--binary",
                    "metalfish",
                    "--output",
                    str(output),
                    "--json-output",
                    str(manifest),
                    "--package-kind",
                    "synthetic",
                    "--file",
                    str(binary),
                ]
            ):
                expect("portable manifest", portable_manifest.main() == 0)
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        data = json.loads(manifest.read_text(encoding="utf-8"))
        expect(
            "actual checkout commit",
            data["package"]["source_commit"]
            == portable_manifest.git_value(["rev-parse", "HEAD"]),
        )
        expect("head branch", data["package"]["source_branch"] == "cuda-support")


def main() -> int:
    test_checker_writes_manifest()
    test_checker_rejects_missing_wdl()
    test_backend_output_compare_accepts_close_outputs()
    test_backend_output_compare_accepts_batched_outputs()
    test_backend_benchmark_compare_writes_summary()
    test_backend_benchmark_compare_checks_selected_release_batch()
    test_backend_benchmark_compare_rejects_selected_release_batch()
    test_backend_benchmark_compare_requires_graph_reuse()
    test_backend_benchmark_compare_allows_expected_without_graph_reuse()
    test_backend_output_compare_accepts_probe_suite()
    test_backend_output_compare_accepts_legacy_scalar_probe_suite()
    test_backend_output_compare_rejects_probe_suite_mismatch()
    test_backend_output_compare_rejects_batched_top_move_drift()
    test_backend_output_compare_rejects_top_move_drift()
    test_probe_suite_runner_writes_multiple_json_probes()
    test_probe_suite_runner_rejects_semantic_drift()
    test_probe_suite_runner_rejects_network_info_drift()
    test_probe_suite_runner_reports_failing_probe_name()
    test_windows_cuda_probe_suite_positions_use_python_default()
    test_windows_cuda_runtime_input_helpers_validate_provenance()
    test_windows_cuda_runtime_input_helpers_select_artifacts()
    test_windows_cuda_runtime_input_manifest_records_file_hashes()
    test_cuda_runtime_input_helpers_validate_complete_zip()
    test_cuda_release_artifact_download_retries_truncated_zip()
    test_windows_cuda_runtime_input_helpers_validate_package_commit()
    test_cuda_runtime_search_contract_paths()
    test_cuda_runtime_observed_parser_extracts_release_facts()
    test_cuda_runtime_manifest_writer_keeps_linux_windows_schema_parity()
    test_cuda_runtime_manifest_requires_timed_mcts_release_artifacts()
    test_cuda_release_artifact_helpers_validate_packages_and_manifests()
    test_cuda_package_validator_rejects_unmanifested_archive_entries()
    test_cuda_release_artifacts_promote_direct_runtime_root()
    test_cuda_release_artifacts_rejects_direct_output_inside_input()
    test_cuda_release_artifacts_reject_direct_runtime_single_target()
    test_cuda_release_artifacts_reject_direct_runtime_commit_drift()
    test_cuda_release_dispatch_direct_root_uses_manifest_sha_default()
    test_cuda_release_dispatch_validates_explicit_run_ids()
    test_cuda_release_artifacts_accept_split_direct_runtime_manifest()
    test_direct_runtime_manifest_merge_preserves_split_targets()
    test_direct_runtime_clears_only_selected_artifact_dir()
    test_cuda_release_artifact_helpers_reject_failed_runtime()
    test_cuda_runtime_manifest_rejects_head_sha_drift()
    test_cuda_runtime_manifest_validates_artifact_files()
    test_cuda_runtime_manifest_rejects_artifacts_outside_root()
    test_cuda_runtime_manifest_enforces_release_policy()
    test_cuda_package_validator_rejects_source_commit_drift()
    test_cuda_package_validator_rejects_hash_drift()
    test_cuda_release_artifact_helpers_require_metal_compare()
    test_cuda_runtime_manifest_requires_benchmark_compare()
    test_network_downloader_validates_gzip_weights()
    test_network_downloader_retries_truncated_gzip()
    test_network_downloader_uses_validated_cache_before_network()
    test_network_downloader_writes_valid_download_to_cache()
    test_gcp_cuda_resource_audit_filters_instances()
    test_portable_manifest_uses_checked_out_commit()
    print("NN backend artifact tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
