#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import lc0_coreml_value_export as lc0_export


def load_mlx() -> Any:
    try:
        import mlx.core as mx
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "mlx is required. Use an isolated venv, for example: "
            "python3.11 -m venv /tmp/metalfish-ane-coremltools-venv && "
            "/tmp/metalfish-ane-coremltools-venv/bin/python -m pip install mlx"
        ) from exc
    return mx


def device_from_name(mx: Any, name: str) -> Any:
    return {"cpu": mx.cpu, "gpu": mx.gpu}[name]


def percentile(values: list[float], pct: float) -> float:
    return lc0_export.percentile(values, pct)


def as_mx(mx: Any, value: Any, dtype: Any) -> Any:
    return mx.array(value.astype(np.float32), dtype=dtype)


def dense_weight(layer: Any, rows: int) -> Any:
    return lc0_export.dense_weight(np, layer, rows)


def decode_layer(layer: Any) -> Any:
    return lc0_export.decode_layer(np, layer)


def mish(mx: Any, x: Any) -> Any:
    return x * mx.tanh(mx.log(1.0 + mx.exp(x)))


def layer_norm(mx: Any, x: Any, gamma: Any, beta: Any, epsilon: float) -> Any:
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
    return (x - mean) * mx.rsqrt(var + epsilon) * gamma + beta


def load_t1_params(mx: Any, weights_path: Path, dtype: Any) -> dict[str, Any]:
    net = lc0_export.load_weights_file(weights_path)
    info = lc0_export.inspect_t1(net)
    if info["encoder_layers"] < 1 or info["ip_emb_channels"] != 256:
        raise RuntimeError("this experimental MLX benchmark currently expects T1-256")

    weights = net.weights
    channels = len(decode_layer(weights.ip_emb_b))
    heads = int(weights.headcount)
    head_dim = channels // heads
    params: dict[str, Any] = {
        "info": info,
        "layers": len(weights.encoder),
        "channels": channels,
        "heads": heads,
        "head_dim": head_dim,
        "alpha": (2.0 * len(weights.encoder)) ** -0.25,
        "pos": as_mx(mx, lc0_export.parse_pos_encoding(np), dtype),
        "ip_emb_w": as_mx(mx, dense_weight(weights.ip_emb_w, channels), dtype),
        "ip_emb_b": as_mx(mx, decode_layer(weights.ip_emb_b), dtype),
        "mult_gate": as_mx(
            mx, decode_layer(weights.ip_mult_gate).reshape(channels, 64).T, dtype
        ),
        "add_gate": as_mx(
            mx, decode_layer(weights.ip_add_gate).reshape(channels, 64).T, dtype
        ),
        "encoders": [],
    }

    for enc in weights.encoder:
        params["encoders"].append(
            {
                "q_w": as_mx(mx, dense_weight(enc.mha.q_w, channels), dtype),
                "q_b": as_mx(mx, decode_layer(enc.mha.q_b), dtype),
                "k_w": as_mx(mx, dense_weight(enc.mha.k_w, channels), dtype),
                "k_b": as_mx(mx, decode_layer(enc.mha.k_b), dtype),
                "v_w": as_mx(mx, dense_weight(enc.mha.v_w, channels), dtype),
                "v_b": as_mx(mx, decode_layer(enc.mha.v_b), dtype),
                "dense_w": as_mx(mx, dense_weight(enc.mha.dense_w, channels), dtype),
                "dense_b": as_mx(mx, decode_layer(enc.mha.dense_b), dtype),
                "ln1_g": as_mx(mx, decode_layer(enc.ln1_gammas), dtype),
                "ln1_b": as_mx(mx, decode_layer(enc.ln1_betas), dtype),
                "ffn1_w": as_mx(mx, dense_weight(enc.ffn.dense1_w, 4 * channels), dtype),
                "ffn1_b": as_mx(mx, decode_layer(enc.ffn.dense1_b), dtype),
                "ffn2_w": as_mx(mx, dense_weight(enc.ffn.dense2_w, channels), dtype),
                "ffn2_b": as_mx(mx, decode_layer(enc.ffn.dense2_b), dtype),
                "ln2_g": as_mx(mx, decode_layer(enc.ln2_gammas), dtype),
                "ln2_b": as_mx(mx, decode_layer(enc.ln2_betas), dtype),
            }
        )

    value_embed_b = decode_layer(weights.ip_val_b)
    value_fc1_b = decode_layer(weights.ip1_val_b)
    value_fc2_b = decode_layer(weights.ip2_val_b)
    params.update(
        {
            "value_embed_w": as_mx(
                mx, dense_weight(weights.ip_val_w, value_embed_b.size), dtype
            ),
            "value_embed_b": as_mx(mx, value_embed_b, dtype),
            "value_fc1_w": as_mx(
                mx, dense_weight(weights.ip1_val_w, value_fc1_b.size), dtype
            ),
            "value_fc1_b": as_mx(mx, value_fc1_b, dtype),
            "value_fc2_w": as_mx(
                mx, dense_weight(weights.ip2_val_w, value_fc2_b.size), dtype
            ),
            "value_fc2_b": as_mx(mx, value_fc2_b, dtype),
            "value_channels": value_embed_b.size,
        }
    )
    return params


def linear(x: Any, weight: Any, bias: Any) -> Any:
    return x @ weight.T + bias


def forward(mx: Any, x: Any, params: dict[str, Any]) -> Any:
    body = mx.concatenate([x, mx.expand_dims(params["pos"], axis=0)], axis=-1)
    body = mish(mx, linear(body, params["ip_emb_w"], params["ip_emb_b"]))
    body = body * params["mult_gate"] + params["add_gate"]

    heads = params["heads"]
    head_dim = params["head_dim"]
    channels = params["channels"]
    scale = 1.0 / math.sqrt(head_dim)
    for enc in params["encoders"]:
        q = linear(body, enc["q_w"], enc["q_b"])
        k = linear(body, enc["k_w"], enc["k_b"])
        v = linear(body, enc["v_w"], enc["v_b"])
        q = mx.transpose(mx.reshape(q, (1, 64, heads, head_dim)), (0, 2, 1, 3))
        k = mx.transpose(mx.reshape(k, (1, 64, heads, head_dim)), (0, 2, 3, 1))
        v = mx.transpose(mx.reshape(v, (1, 64, heads, head_dim)), (0, 2, 1, 3))
        attn = mx.softmax((q @ k) * scale, axis=-1) @ v
        attn = mx.reshape(mx.transpose(attn, (0, 2, 1, 3)), (1, 64, channels))
        attn = linear(attn, enc["dense_w"], enc["dense_b"])
        body = layer_norm(
            mx,
            body + attn * params["alpha"],
            enc["ln1_g"],
            enc["ln1_b"],
            1e-6,
        )

        ffn = mish(mx, linear(body, enc["ffn1_w"], enc["ffn1_b"]))
        ffn = linear(ffn, enc["ffn2_w"], enc["ffn2_b"])
        body = layer_norm(
            mx,
            body + ffn * params["alpha"],
            enc["ln2_g"],
            enc["ln2_b"],
            1e-6,
        )

    value = mish(mx, linear(body, params["value_embed_w"], params["value_embed_b"]))
    value = mx.reshape(value, (1, 64 * params["value_channels"]))
    value = mish(mx, linear(value, params["value_fc1_w"], params["value_fc1_b"]))
    return mx.softmax(linear(value, params["value_fc2_w"], params["value_fc2_b"]), axis=-1)


def benchmark(mx: Any, params: dict[str, Any], dtype: Any, warmup: int, iterations: int) -> dict[str, Any]:
    sample = mx.zeros((1, 64, 112), dtype=dtype)
    for _ in range(warmup):
        out = forward(mx, sample, params)
        mx.eval(out)

    latencies: list[float] = []
    out = None
    for _ in range(iterations):
        start = time.perf_counter()
        out = forward(mx, sample, params)
        mx.eval(out)
        latencies.append((time.perf_counter() - start) * 1000.0)

    output = np.array(out, copy=False).tolist() if out is not None else []
    return {
        "iterations": iterations,
        "median_ms": statistics.median(latencies),
        "mean_ms": statistics.fmean(latencies),
        "p90_ms": percentile(latencies, 0.90),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "output": output,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    mx = load_mlx()
    mx.set_default_device(device_from_name(mx, args.device))
    dtype = {"fp16": mx.float16, "fp32": mx.float32}[args.precision]
    params = load_t1_params(mx, Path(args.weights), dtype)
    result = {
        "weights": str(Path(args.weights).resolve()),
        "device": args.device,
        "precision": args.precision,
        "network": params["info"],
        "mlx_version": getattr(mx, "__version__", "unknown"),
    }
    if args.benchmark:
        result["benchmark"] = benchmark(mx, params, dtype, args.warmup, args.iterations)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental MLX value-head benchmark for the Lc0 T1-256 net."
    )
    parser.add_argument("weights", help="Lc0 .pb or .pb.gz weights")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("MetalFish Lc0 MLX value benchmark")
        print(f"  Weights:   {result['weights']}")
        print(f"  Device:    {result['device']}")
        print(f"  Precision: {result['precision']}")
        print(f"  MLX:       {result['mlx_version']}")
        if "benchmark" in result:
            bench = result["benchmark"]
            print(
                "  Predict:   "
                f"median={bench['median_ms']:.3f} ms "
                f"mean={bench['mean_ms']:.3f} ms p90={bench['p90_ms']:.3f} ms"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
