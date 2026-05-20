#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import lc0_coreml_value_export as exporter  # noqa: E402


def expect(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)


def test_percentile_uses_sorted_values() -> None:
    expect("empty", exporter.percentile([], 0.5) == 0.0)
    expect("p0", exporter.percentile([3.0, 1.0, 2.0], 0.0) == 1.0)
    expect("p90", exporter.percentile([3.0, 1.0, 2.0], 0.9) == 3.0)


def test_compute_unit_mapping() -> None:
    fake_ct = types.SimpleNamespace(
        ComputeUnit=types.SimpleNamespace(
            ALL="all", CPU_ONLY="cpu", CPU_AND_GPU="cpu-gpu", CPU_AND_NE="cpu-ne"
        )
    )
    expect("cpu", exporter.compute_unit_from_name(fake_ct, "cpu") == "cpu")
    expect("cpu-ne", exporter.compute_unit_from_name(fake_ct, "cpu-ne") == "cpu-ne")


def test_precision_mapping() -> None:
    fake_ct = types.SimpleNamespace(
        precision=types.SimpleNamespace(FLOAT16="fp16", FLOAT32="fp32")
    )
    expect("fp16", exporter.precision_from_name(fake_ct, "fp16") == "fp16")
    expect("fp32", exporter.precision_from_name(fake_ct, "fp32") == "fp32")


def test_parse_args() -> None:
    args = exporter.parse_args(
        [
            "networks/t1-256x10-distilled-swa-2432500.pb.gz",
            "--compute-unit",
            "cpu-ne",
            "--precision",
            "fp32",
            "--benchmark",
            "--warmup",
            "2",
            "--iterations",
            "7",
            "--value-head-fp32",
            "--policy-head-fp32",
            "--output-stage",
            "heads",
            "--encoder-layers",
            "3",
            "--batch-size",
            "8",
        ]
    )
    expect("weights parsed", args.weights.endswith(".pb.gz"))
    expect("compute parsed", args.compute_unit == "cpu-ne")
    expect("precision parsed", args.precision == "fp32")
    expect("benchmark parsed", args.benchmark)
    expect("value head fp32 parsed", args.value_head_fp32)
    expect("policy head fp32 parsed", args.policy_head_fp32)
    expect("warmup parsed", args.warmup == 2)
    expect("iterations parsed", args.iterations == 7)
    expect("output stage parsed", args.output_stage == "heads")
    expect("encoder layers parsed", args.encoder_layers == 3)
    expect("batch size parsed", args.batch_size == 8)


def test_prediction_summary() -> None:
    fake_np = types.SimpleNamespace(
        asarray=lambda value, dtype=None: value.astype(dtype),
        float32="float32",
    )
    import numpy as real_np

    output = {"x": real_np.array([[1.0, 2.0, 3.0]], dtype=real_np.float32)}
    summary = exporter.summarize_prediction_outputs(fake_np, output)
    expect("summary shape", summary["x"]["shape"] == [1, 3])
    expect("summary mean", summary["x"]["mean"] == 2.0)
    expect("summary first", summary["x"]["first8"] == [1.0, 2.0, 3.0])


def test_attention_policy_map() -> None:
    import numpy as real_np

    indices = exporter.attention_policy_gather_indices(real_np)
    expect("policy index count", indices.shape == (1858,))
    expect("policy index min", int(indices.min()) >= 0)
    expect("policy index max", int(indices.max()) < 64 * 64 + 8 * 24)


def test_effective_activation_fixup() -> None:
    class FakeWeights:
        encoder = [object()]

        @staticmethod
        def HasField(name: str) -> bool:
            return name == "smolgen_w"

    fake_net = types.SimpleNamespace(
        format=types.SimpleNamespace(
            network_format=types.SimpleNamespace(
                default_activation=1,
                smolgen_activation=0,
                ffn_activation=0,
                network=4,
            )
        ),
        weights=FakeWeights(),
    )
    activations = exporter.effective_activations(fake_net)
    expect("default mish", activations["default"] == "mish")
    expect("smolgen swish", activations["smolgen"] == "swish")
    expect("ffn relu2", activations["ffn"] == "relu_2")


def main() -> int:
    test_percentile_uses_sorted_values()
    test_compute_unit_mapping()
    test_precision_mapping()
    test_parse_args()
    test_prediction_summary()
    test_attention_policy_map()
    test_effective_activation_fixup()
    print("Lc0 Core ML value exporter tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
