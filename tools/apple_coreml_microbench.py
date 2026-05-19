#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def load_coremltools() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "coremltools and numpy are required. Use a temporary venv, for "
            "example: python3.11 -m venv /tmp/metalfish-coremltools-venv && "
            "/tmp/metalfish-coremltools-venv/bin/python -m pip install "
            "coremltools==9.0"
        ) from exc
    return np, ct, mb, types


def compute_unit_from_name(ct: Any, name: str) -> Any:
    units = {
        "all": ct.ComputeUnit.ALL,
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu-gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu-ne": ct.ComputeUnit.CPU_AND_NE,
    }
    return units[name]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[idx]


def build_dense_model(
    np: Any,
    ct: Any,
    mb: Any,
    types: Any,
    batch: int,
    inputs: int,
    outputs: int,
    compute_unit: Any,
    seed: int,
) -> tuple[Any, Any, Any]:
    rng = np.random.default_rng(seed)
    weight = rng.normal(0.0, 0.02, size=(outputs, inputs)).astype(np.float32)
    bias = rng.normal(0.0, 0.01, size=(outputs,)).astype(np.float32)

    @mb.program(input_specs=[mb.TensorSpec(shape=(batch, inputs), dtype=types.fp32)])
    def program(x):  # type: ignore[no-untyped-def]
        w = mb.const(val=weight)
        b = mb.const(val=bias)
        return mb.linear(x=x, weight=w, bias=b)

    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_units=compute_unit,
    )
    return model, weight, bias


def build_transformer_model(
    np: Any,
    ct: Any,
    mb: Any,
    types: Any,
    batch: int,
    tokens: int,
    channels: int,
    heads: int,
    ffn_channels: int,
    layers: int,
    activation: str,
    compute_unit: Any,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    params = make_transformer_params(
        np, channels, heads, ffn_channels, layers, activation, seed
    )
    head_dim = params["head_dim"]

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(batch, tokens, channels), dtype=types.fp32)]
    )
    def program(x):  # type: ignore[no-untyped-def]
        layer = x
        for _ in range(layers):
            layer = transformer_block_mb(
                mb, np, layer, params, batch, tokens, channels, heads, head_dim, activation
            )
        return layer

    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_units=compute_unit,
    )
    return model, params


def make_transformer_params(
    np: Any,
    channels: int,
    heads: int,
    ffn_channels: int,
    layers: int,
    activation: str,
    seed: int,
) -> dict[str, Any]:
    if channels % heads != 0:
        raise RuntimeError("transformer channels must be divisible by heads")
    head_dim = channels // heads
    rng = np.random.default_rng(seed)
    return {
        "ln1_gamma": np.ones((channels,), dtype=np.float32),
        "ln1_beta": np.zeros((channels,), dtype=np.float32),
        "q_w": rng.normal(0.0, 0.02, size=(channels, channels)).astype(np.float32),
        "q_b": np.zeros((channels,), dtype=np.float32),
        "k_w": rng.normal(0.0, 0.02, size=(channels, channels)).astype(np.float32),
        "k_b": np.zeros((channels,), dtype=np.float32),
        "v_w": rng.normal(0.0, 0.02, size=(channels, channels)).astype(np.float32),
        "v_b": np.zeros((channels,), dtype=np.float32),
        "o_w": rng.normal(0.0, 0.02, size=(channels, channels)).astype(np.float32),
        "o_b": np.zeros((channels,), dtype=np.float32),
        "ln2_gamma": np.ones((channels,), dtype=np.float32),
        "ln2_beta": np.zeros((channels,), dtype=np.float32),
        "ffn1_w": rng.normal(0.0, 0.02, size=(ffn_channels, channels)).astype(
            np.float32
        ),
        "ffn1_b": np.zeros((ffn_channels,), dtype=np.float32),
        "ffn2_w": rng.normal(0.0, 0.02, size=(channels, ffn_channels)).astype(
            np.float32
        ),
        "ffn2_b": np.zeros((channels,), dtype=np.float32),
        "heads": heads,
        "head_dim": head_dim,
        "layers": layers,
        "activation": activation,
    }


def build_toy_network_model(
    np: Any,
    ct: Any,
    mb: Any,
    types: Any,
    batch: int,
    tokens: int,
    input_channels: int,
    channels: int,
    heads: int,
    ffn_channels: int,
    layers: int,
    activation: str,
    policy_channels: int,
    value_outputs: int,
    compute_unit: Any,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    params = make_transformer_params(
        np, channels, heads, ffn_channels, layers, activation, seed + 100
    )
    params.update(
        {
            "embed_w": rng.normal(0.0, 0.02, size=(channels, input_channels)).astype(
                np.float32
            ),
            "embed_b": np.zeros((channels,), dtype=np.float32),
            "policy_w": rng.normal(
                0.0, 0.02, size=(policy_channels, channels)
            ).astype(np.float32),
            "policy_b": np.zeros((policy_channels,), dtype=np.float32),
            "value_w": rng.normal(0.0, 0.02, size=(value_outputs, channels)).astype(
                np.float32
            ),
            "value_b": np.zeros((value_outputs,), dtype=np.float32),
            "input_channels": input_channels,
            "policy_channels": policy_channels,
            "value_outputs": value_outputs,
        }
    )
    head_dim = params["head_dim"]

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(batch, tokens, input_channels), dtype=types.fp32)
        ]
    )
    def program(x):  # type: ignore[no-untyped-def]
        layer = mb.linear(
            x=x,
            weight=mb.const(val=params["embed_w"]),
            bias=mb.const(val=params["embed_b"]),
        )
        for _ in range(layers):
            layer = transformer_block_mb(
                mb, np, layer, params, batch, tokens, channels, heads, head_dim, activation
            )
        policy = mb.linear(
            x=layer,
            weight=mb.const(val=params["policy_w"]),
            bias=mb.const(val=params["policy_b"]),
        )
        pooled = mb.reduce_mean(x=layer, axes=[1], keep_dims=False)
        value = mb.linear(
            x=pooled,
            weight=mb.const(val=params["value_w"]),
            bias=mb.const(val=params["value_b"]),
        )
        return policy, value

    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_units=compute_unit,
    )
    return model, params


def transformer_block_mb(
    mb: Any,
    np: Any,
    x: Any,
    params: dict[str, Any],
    batch: int,
    tokens: int,
    channels: int,
    heads: int,
    head_dim: int,
    activation: str,
) -> Any:
    ln1_gamma = mb.const(val=params["ln1_gamma"])
    ln1_beta = mb.const(val=params["ln1_beta"])
    y = mb.layer_norm(
        x=x, axes=[-1], gamma=ln1_gamma, beta=ln1_beta, epsilon=1e-5
    )

    q = mb.linear(
        x=y, weight=mb.const(val=params["q_w"]), bias=mb.const(val=params["q_b"])
    )
    k = mb.linear(
        x=y, weight=mb.const(val=params["k_w"]), bias=mb.const(val=params["k_b"])
    )
    v = mb.linear(
        x=y, weight=mb.const(val=params["v_w"]), bias=mb.const(val=params["v_b"])
    )

    q = mb.reshape(x=q, shape=[batch, tokens, heads, head_dim])
    k = mb.reshape(x=k, shape=[batch, tokens, heads, head_dim])
    v = mb.reshape(x=v, shape=[batch, tokens, heads, head_dim])
    q = mb.transpose(x=q, perm=[0, 2, 1, 3])
    k = mb.transpose(x=k, perm=[0, 2, 3, 1])
    v = mb.transpose(x=v, perm=[0, 2, 1, 3])
    scores = mb.matmul(x=q, y=k)
    scale = mb.const(val=np.array(1.0 / np.sqrt(head_dim), dtype=np.float32))
    scores = mb.mul(x=scores, y=scale)
    probs = mb.softmax(x=scores, axis=-1)
    context = mb.matmul(x=probs, y=v)
    context = mb.transpose(x=context, perm=[0, 2, 1, 3])
    context = mb.reshape(x=context, shape=[batch, tokens, channels])
    attn_out = mb.linear(
        x=context,
        weight=mb.const(val=params["o_w"]),
        bias=mb.const(val=params["o_b"]),
    )
    residual = mb.add(x=x, y=attn_out)

    ln2_gamma = mb.const(val=params["ln2_gamma"])
    ln2_beta = mb.const(val=params["ln2_beta"])
    z = mb.layer_norm(
        x=residual, axes=[-1], gamma=ln2_gamma, beta=ln2_beta, epsilon=1e-5
    )
    z = mb.linear(
        x=z,
        weight=mb.const(val=params["ffn1_w"]),
        bias=mb.const(val=params["ffn1_b"]),
    )
    if activation == "swish":
        z = mb.mul(x=z, y=mb.sigmoid(x=z))
    else:
        z = mb.gelu(x=z, mode="TANH_APPROXIMATION")
    z = mb.linear(
        x=z,
        weight=mb.const(val=params["ffn2_w"]),
        bias=mb.const(val=params["ffn2_b"]),
    )
    return mb.add(x=residual, y=z)


def benchmark_predict(model: Any, sample: Any, warmup: int, iterations: int) -> dict[str, Any]:
    for _ in range(warmup):
        model.predict({"x": sample})

    latencies_ms: list[float] = []
    last_output = None
    for _ in range(iterations):
        start = time.perf_counter()
        last_output = model.predict({"x": sample})
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "iterations": iterations,
        "median_ms": statistics.median(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms),
        "p90_ms": percentile(latencies_ms, 0.90),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "output_name": next(iter(last_output.keys())) if last_output else "",
    }


def benchmark_numpy(
    np: Any,
    sample: Any,
    weight: Any,
    bias: Any,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        sample @ weight.T + bias

    latencies_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        sample @ weight.T + bias
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "iterations": iterations,
        "median_ms": statistics.median(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms),
        "p90_ms": percentile(latencies_ms, 0.90),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def layer_norm_np(np: Any, x: Any, gamma: Any, beta: Any, epsilon: float = 1e-5) -> Any:
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + epsilon) * gamma + beta


def softmax_np(np: Any, x: Any, axis: int = -1) -> Any:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def gelu_tanh_np(np: Any, x: Any) -> Any:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def transformer_numpy_once(np: Any, sample: Any, params: dict[str, Any]) -> Any:
    heads = params["heads"]
    head_dim = params["head_dim"]
    batch, tokens, channels = sample.shape
    layer = sample

    for _ in range(params["layers"]):
        y = layer_norm_np(np, layer, params["ln1_gamma"], params["ln1_beta"])
        q = y @ params["q_w"].T + params["q_b"]
        k = y @ params["k_w"].T + params["k_b"]
        v = y @ params["v_w"].T + params["v_b"]
        q = q.reshape(batch, tokens, heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, tokens, heads, head_dim).transpose(0, 2, 3, 1)
        v = v.reshape(batch, tokens, heads, head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k) * (1.0 / np.sqrt(head_dim))
        probs = softmax_np(np, scores, axis=-1)
        context = (probs @ v).transpose(0, 2, 1, 3).reshape(batch, tokens, channels)
        residual = layer + context @ params["o_w"].T + params["o_b"]

        z = layer_norm_np(np, residual, params["ln2_gamma"], params["ln2_beta"])
        z = z @ params["ffn1_w"].T + params["ffn1_b"]
        if params["activation"] == "swish":
            z = z / (1.0 + np.exp(-z))
        else:
            z = gelu_tanh_np(np, z)
        z = z @ params["ffn2_w"].T + params["ffn2_b"]
        layer = residual + z
    return layer


def toy_network_numpy_once(np: Any, sample: Any, params: dict[str, Any]) -> tuple[Any, Any]:
    layer = sample @ params["embed_w"].T + params["embed_b"]
    layer = transformer_numpy_once(np, layer, params)
    policy = layer @ params["policy_w"].T + params["policy_b"]
    pooled = np.mean(layer, axis=1)
    value = pooled @ params["value_w"].T + params["value_b"]
    return policy, value


def benchmark_numpy_transformer(
    np: Any,
    sample: Any,
    params: dict[str, Any],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        transformer_numpy_once(np, sample, params)

    latencies_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        transformer_numpy_once(np, sample, params)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "iterations": iterations,
        "median_ms": statistics.median(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms),
        "p90_ms": percentile(latencies_ms, 0.90),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def benchmark_numpy_toy_network(
    np: Any,
    sample: Any,
    params: dict[str, Any],
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        toy_network_numpy_once(np, sample, params)

    latencies_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        toy_network_numpy_once(np, sample, params)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "iterations": iterations,
        "median_ms": statistics.median(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms),
        "p90_ms": percentile(latencies_ms, 0.90),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    np, ct, mb, types = load_coremltools()
    compute_unit = compute_unit_from_name(ct, args.compute_unit)
    if args.layers < 1:
        raise RuntimeError("--layers must be at least 1")

    if args.model == "dense":
        sample = np.random.default_rng(args.seed + 1).normal(
            0.0, 1.0, size=(args.batch, args.inputs)
        ).astype(np.float32)
    elif args.model == "transformer":
        sample = np.random.default_rng(args.seed + 1).normal(
            0.0, 1.0, size=(args.batch, args.tokens, args.channels)
        ).astype(np.float32)
    else:
        sample = np.random.default_rng(args.seed + 1).normal(
            0.0, 1.0, size=(args.batch, args.tokens, args.input_channels)
        ).astype(np.float32)

    build_start = time.perf_counter()
    if args.model == "dense":
        model, weight, bias = build_dense_model(
            np,
            ct,
            mb,
            types,
            args.batch,
            args.inputs,
            args.outputs,
            compute_unit,
            args.seed,
        )
        numpy_result = benchmark_numpy(
            np, sample, weight, bias, args.warmup, args.iterations
        )
    elif args.model == "transformer":
        model, params = build_transformer_model(
            np,
            ct,
            mb,
            types,
            args.batch,
            args.tokens,
            args.channels,
            args.heads,
            args.channels * args.ffn_mult,
            args.layers,
            args.activation,
            compute_unit,
            args.seed,
        )
        numpy_result = benchmark_numpy_transformer(
            np, sample, params, args.warmup, args.iterations
        )
    else:
        model, params = build_toy_network_model(
            np,
            ct,
            mb,
            types,
            args.batch,
            args.tokens,
            args.input_channels,
            args.channels,
            args.heads,
            args.channels * args.ffn_mult,
            args.layers,
            args.activation,
            args.policy_channels,
            args.value_outputs,
            compute_unit,
            args.seed,
        )
        numpy_result = benchmark_numpy_toy_network(
            np, sample, params, args.warmup, args.iterations
        )
    build_ms = (time.perf_counter() - build_start) * 1000.0

    package_path = ""
    if args.save_model:
        package_path = str(Path(args.save_model).resolve())
        model.save(package_path)
    else:
        with tempfile.TemporaryDirectory(prefix="metalfish-coreml-microbench-") as tmp:
            package_path = str(Path(tmp) / "DenseMicrobench.mlpackage")
            model.save(package_path)

    return {
        "model": args.model,
        "batch": args.batch,
        "inputs": args.inputs,
        "outputs": args.outputs,
        "tokens": args.tokens,
        "input_channels": args.input_channels,
        "channels": args.channels,
        "heads": args.heads,
        "ffn_mult": args.ffn_mult,
        "layers": args.layers,
        "activation": args.activation,
        "policy_channels": args.policy_channels,
        "value_outputs": args.value_outputs,
        "compute_unit": args.compute_unit,
        "build_ms": build_ms,
        "model_package": package_path if args.save_model else "",
        "coreml": benchmark_predict(model, sample, args.warmup, args.iterations),
        "numpy": numpy_result,
    }


def print_human(result: dict[str, Any]) -> None:
    print("MetalFish Core ML microbenchmark")
    print(f"  Model:        {result['model']}")
    if result["model"] == "dense":
        print(
            f"  Shape:        batch={result['batch']} "
            f"inputs={result['inputs']} outputs={result['outputs']}"
        )
    elif result["model"] == "transformer":
        print(
            f"  Shape:        batch={result['batch']} tokens={result['tokens']} "
            f"channels={result['channels']} heads={result['heads']} "
            f"ffn_mult={result['ffn_mult']} layers={result['layers']} "
            f"activation={result['activation']}"
        )
    else:
        print(
            f"  Shape:        batch={result['batch']} tokens={result['tokens']} "
            f"input_channels={result['input_channels']} "
            f"channels={result['channels']} heads={result['heads']} "
            f"ffn_mult={result['ffn_mult']} layers={result['layers']} "
            f"policy_channels={result['policy_channels']} "
            f"value_outputs={result['value_outputs']} "
            f"activation={result['activation']}"
        )
    print(f"  Compute unit: {result['compute_unit']}")
    print(f"  Build time:   {result['build_ms']:.3f} ms")
    coreml = result["coreml"]
    numpy = result["numpy"]
    print(
        "  Core ML:      "
        f"median={coreml['median_ms']:.4f} ms "
        f"mean={coreml['mean_ms']:.4f} ms "
        f"p90={coreml['p90_ms']:.4f} ms"
    )
    print(
        "  NumPy:        "
        f"median={numpy['median_ms']:.4f} ms "
        f"mean={numpy['mean_ms']:.4f} ms "
        f"p90={numpy['p90_ms']:.4f} ms"
    )
    if result["model_package"]:
        print(f"  Saved model:  {result['model_package']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Core ML kernels for Apple accelerator experiments."
    )
    parser.add_argument(
        "--model", choices=["dense", "transformer", "toy-network"], default="dense"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--inputs", type=int, default=32)
    parser.add_argument("--outputs", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--input-channels", type=int, default=112)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--activation", choices=["gelu", "swish"], default="gelu")
    parser.add_argument("--policy-channels", type=int, default=32)
    parser.add_argument("--value-outputs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--compute-unit",
        choices=["all", "cpu", "cpu-gpu", "cpu-ne"],
        default="cpu-ne",
    )
    parser.add_argument("--save-model", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_benchmark(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
