#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
from contextlib import contextmanager

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import check_nn_backend_artifacts as checker  # noqa: E402
from tools import compare_nn_backend_outputs as comparer  # noqa: E402


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
        f"backend: {backend_label}\n    batches: b1=1.0ms checksum=0\n",
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
    top_moves: list[str] | None = None,
    value: float = 0.25,
    policy_shift: float = 0.0,
) -> None:
    moves = top_moves or ["e2e4", "d2d4", "g1f3"]
    policy = [1.0 + policy_shift, 0.5, -0.25, -1.0]
    path.write_text(
        "info string warmup\n"
        + json.dumps(
            {
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "backend": backend,
                "network_info": f"{label} synthetic",
                "transform": 0,
                "value": value,
                "has_wdl": True,
                "wdl": [0.2, 0.7, 0.1],
                "has_moves_left": True,
                "moves_left": 12.5,
                "policy_top": [
                    {"move": moves[0], "logit": 1.0 + policy_shift},
                    {"move": moves[1], "logit": 0.5},
                    {"move": moves[2], "logit": -0.25},
                ],
                "policy": policy,
            }
        )
        + "\n",
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


def main() -> int:
    test_checker_writes_manifest()
    test_checker_rejects_missing_wdl()
    test_backend_output_compare_accepts_close_outputs()
    test_backend_output_compare_rejects_top_move_drift()
    print("NN backend artifact tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
