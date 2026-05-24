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


def main() -> int:
    test_checker_writes_manifest()
    test_checker_rejects_missing_wdl()
    print("NN backend artifact tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
