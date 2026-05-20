#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
PROTO = ROOT / "src" / "nn" / "proto" / "net.proto"
POS_TABLE = ROOT / "src" / "nn" / "metal" / "tables" / "attention_policy_map.h"


def load_coremltools() -> tuple[Any, Any, Any, Any]:
    try:
        import coremltools as ct
        import numpy as np
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "coremltools, numpy, and protobuf are required. Use an isolated "
            "venv, for example: python3.11 -m venv /tmp/metalfish-ane-coremltools-venv "
            "&& /tmp/metalfish-ane-coremltools-venv/bin/python -m pip install "
            "coremltools==9.0 'numpy<3'"
        ) from exc
    return np, ct, mb, types


def compute_unit_from_name(ct: Any, name: str) -> Any:
    return {
        "all": ct.ComputeUnit.ALL,
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu-gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu-ne": ct.ComputeUnit.CPU_AND_NE,
    }[name]


def precision_from_name(ct: Any, name: str) -> Any:
    return {
        "fp16": ct.precision.FLOAT16,
        "fp32": ct.precision.FLOAT32,
    }[name]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[idx]


def load_net_pb2() -> Any:
    protoc = shutil.which("protoc")
    if not protoc:
        raise RuntimeError("protoc is required to generate Python bindings for net.proto")
    tmp = tempfile.TemporaryDirectory(prefix="metalfish-lc0-proto-")
    out_dir = Path(tmp.name)
    result = subprocess.run(
        [protoc, f"-I{ROOT}", f"--python_out={out_dir}", str(PROTO.relative_to(ROOT))],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        tmp.cleanup()
        raise RuntimeError(result.stderr.strip() or "protoc failed")

    module_path = out_dir / "src" / "nn" / "proto" / "net_pb2.py"
    spec = importlib.util.spec_from_file_location("metalfish_lc0_net_pb2", module_path)
    if spec is None or spec.loader is None:
        tmp.cleanup()
        raise RuntimeError("failed to import generated net_pb2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._metalfish_tmpdir = tmp
    return module


def read_weights_bytes(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return f.read()
    return path.read_bytes()


def load_weights_file(path: Path) -> Any:
    net_pb2 = load_net_pb2()
    net = net_pb2.Net()
    if not net.ParseFromString(read_weights_bytes(path)):
        raise RuntimeError(f"failed to parse {path}")
    if net.magic != 0x1C0:
        raise RuntimeError(f"{path} is not an Lc0 weights file")
    return net


def decode_layer(np: Any, layer: Any) -> Any:
    raw = layer.params
    if not raw:
        return np.zeros((0,), dtype=np.float32)

    encoding = layer.encoding
    if encoding == 0:
        encoding = layer.LINEAR16

    if encoding == layer.FLOAT32:
        return np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if encoding == layer.FLOAT16:
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if encoding == layer.BFLOAT16:
        values = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return values.view(np.float32)
    if encoding == layer.LINEAR16:
        values = np.frombuffer(raw, dtype="<u2").astype(np.float32)
        return layer.min_val + (values / 65535.0) * (layer.max_val - layer.min_val)
    raise RuntimeError(f"unsupported layer encoding {encoding}")


def dense_weight(np: Any, layer: Any, rows: int | None = None) -> Any:
    values = decode_layer(np, layer)
    if rows is None:
        dims = list(layer.dims)
        if len(dims) == 2:
            rows = int(dims[0])
        else:
            raise RuntimeError("dense rows must be provided for layers without dims")
    if values.size % rows != 0:
        raise RuntimeError(f"cannot reshape dense layer of {values.size} values to {rows} rows")
    return values.reshape(rows, values.size // rows).astype(np.float32)


def activation_mb(mb: Any, np: Any, x: Any, name: str) -> Any:
    if name == "mish":
        one = mb.const(val=np.array(1.0, dtype=np.float32))
        return mb.mul(x=x, y=mb.tanh(x=mb.log(x=mb.add(x=one, y=mb.exp(x=x)))))
    if name == "swish":
        return mb.mul(x=x, y=mb.sigmoid(x=x))
    if name == "relu":
        return mb.relu(x=x)
    raise RuntimeError(f"unsupported activation {name}")


def layer_norm_mb(
    mb: Any,
    np: Any,
    parent: Any,
    secondary: Any,
    gamma: Any,
    beta: Any,
    alpha: float,
    epsilon: float,
) -> Any:
    if secondary is not None:
        if alpha != 1.0:
            secondary = mb.mul(x=secondary, y=mb.const(val=np.array(alpha, dtype=np.float32)))
        parent = mb.add(x=parent, y=secondary)
    return mb.layer_norm(
        x=parent,
        axes=[-1],
        gamma=mb.const(val=gamma.astype(np.float32)),
        beta=mb.const(val=beta.astype(np.float32)),
        epsilon=epsilon,
    )


def linear_mb(mb: Any, x: Any, weight: Any, bias: Any) -> Any:
    return mb.linear(
        x=x,
        weight=mb.const(val=weight.astype("float32")),
        bias=mb.const(val=bias.astype("float32")),
    )


def parse_pos_encoding(np: Any) -> Any:
    text = POS_TABLE.read_text(encoding="utf-8")
    marker = "const float kPosEncoding[64][kNumPosEncodingChannels] = {"
    start = text.index(marker) + len(marker)
    end = text.index("};", start)
    table_text = text[start:end]
    values = [float(token) for token in table_text.replace("{", " ").replace("}", " ").replace(",", " ").split()]
    if len(values) != 64 * 64:
        raise RuntimeError(f"expected 4096 positional values, found {len(values)}")
    return np.array(values, dtype=np.float32).reshape(64, 64)


def inspect_t1(net: Any) -> dict[str, Any]:
    weights = net.weights
    fmt = net.format.network_format
    return {
        "magic": hex(net.magic),
        "input": fmt.input,
        "output": fmt.output,
        "network": fmt.network,
        "policy": fmt.policy,
        "value": fmt.value,
        "moves_left": fmt.moves_left,
        "default_activation": fmt.default_activation,
        "ffn_activation": fmt.ffn_activation,
        "encoder_layers": len(weights.encoder),
        "headcount": weights.headcount,
        "ip_emb_channels": len(weights.ip_emb_b.params) // 2,
        "value_channels": len(weights.ip_val_b.params) // 2,
    }


def build_value_model(
    np: Any,
    ct: Any,
    mb: Any,
    types: Any,
    net: Any,
    compute_unit: Any,
    compute_precision: Any,
) -> Any:
    weights = net.weights
    layers = len(weights.encoder)
    channels = len(decode_layer(np, weights.ip_emb_b))
    heads = int(weights.headcount)
    head_dim = channels // heads
    alpha = (2.0 * layers) ** -0.25
    pos = parse_pos_encoding(np)

    ip_emb_w = dense_weight(np, weights.ip_emb_w, channels)
    ip_emb_b = decode_layer(np, weights.ip_emb_b)
    mult_gate = decode_layer(np, weights.ip_mult_gate).reshape(channels, 64).T
    add_gate = decode_layer(np, weights.ip_add_gate).reshape(channels, 64).T

    encoders: list[dict[str, Any]] = []
    for enc in weights.encoder:
        encoders.append(
            {
                "q_w": dense_weight(np, enc.mha.q_w, channels),
                "q_b": decode_layer(np, enc.mha.q_b),
                "k_w": dense_weight(np, enc.mha.k_w, channels),
                "k_b": decode_layer(np, enc.mha.k_b),
                "v_w": dense_weight(np, enc.mha.v_w, channels),
                "v_b": decode_layer(np, enc.mha.v_b),
                "dense_w": dense_weight(np, enc.mha.dense_w, channels),
                "dense_b": decode_layer(np, enc.mha.dense_b),
                "ln1_g": decode_layer(np, enc.ln1_gammas),
                "ln1_b": decode_layer(np, enc.ln1_betas),
                "ffn1_w": dense_weight(np, enc.ffn.dense1_w, 4 * channels),
                "ffn1_b": decode_layer(np, enc.ffn.dense1_b),
                "ffn2_w": dense_weight(np, enc.ffn.dense2_w, channels),
                "ffn2_b": decode_layer(np, enc.ffn.dense2_b),
                "ln2_g": decode_layer(np, enc.ln2_gammas),
                "ln2_b": decode_layer(np, enc.ln2_betas),
            }
        )

    value_embed_b = decode_layer(np, weights.ip_val_b)
    value_embed_w = dense_weight(np, weights.ip_val_w, value_embed_b.size)
    value_fc1_b = decode_layer(np, weights.ip1_val_b)
    value_fc1_w = dense_weight(np, weights.ip1_val_w, value_fc1_b.size)
    value_fc2_b = decode_layer(np, weights.ip2_val_b)
    value_fc2_w = dense_weight(np, weights.ip2_val_w, value_fc2_b.size)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 64, 112), dtype=types.fp32)])
    def program(x):  # type: ignore[no-untyped-def]
        pos_const = mb.const(val=pos)
        pos_batch = mb.reshape(x=pos_const, shape=[1, 64, 64])
        body = mb.concat(values=[x, pos_batch], axis=-1)
        body = linear_mb(mb, body, ip_emb_w, ip_emb_b)
        body = activation_mb(mb, np, body, "mish")
        body = mb.mul(x=body, y=mb.const(val=mult_gate.astype(np.float32)))
        body = mb.add(x=body, y=mb.const(val=add_gate.astype(np.float32)))

        for enc in encoders:
            q = linear_mb(mb, body, enc["q_w"], enc["q_b"])
            k = linear_mb(mb, body, enc["k_w"], enc["k_b"])
            v = linear_mb(mb, body, enc["v_w"], enc["v_b"])
            q = mb.reshape(x=q, shape=[1, 64, heads, head_dim])
            k = mb.reshape(x=k, shape=[1, 64, heads, head_dim])
            v = mb.reshape(x=v, shape=[1, 64, heads, head_dim])
            q = mb.transpose(x=q, perm=[0, 2, 1, 3])
            k = mb.transpose(x=k, perm=[0, 2, 3, 1])
            v = mb.transpose(x=v, perm=[0, 2, 1, 3])
            scores = mb.matmul(x=q, y=k)
            scale = mb.const(val=np.array(1.0 / math.sqrt(head_dim), dtype=np.float32))
            scores = mb.mul(x=scores, y=scale)
            probs = mb.softmax(x=scores, axis=-1)
            attn = mb.matmul(x=probs, y=v)
            attn = mb.transpose(x=attn, perm=[0, 2, 1, 3])
            attn = mb.reshape(x=attn, shape=[1, 64, channels])
            attn = linear_mb(mb, attn, enc["dense_w"], enc["dense_b"])
            body = layer_norm_mb(
                mb, np, body, attn, enc["ln1_g"], enc["ln1_b"], alpha, 1e-6
            )

            ffn = linear_mb(mb, body, enc["ffn1_w"], enc["ffn1_b"])
            ffn = activation_mb(mb, np, ffn, "mish")
            ffn = linear_mb(mb, ffn, enc["ffn2_w"], enc["ffn2_b"])
            body = layer_norm_mb(
                mb, np, body, ffn, enc["ln2_g"], enc["ln2_b"], alpha, 1e-6
            )

        value = linear_mb(mb, body, value_embed_w, value_embed_b)
        value = activation_mb(mb, np, value, "mish")
        value = mb.reshape(x=value, shape=[1, 64 * value_embed_b.size])
        value = linear_mb(mb, value, value_fc1_w, value_fc1_b)
        value = activation_mb(mb, np, value, "mish")
        value = linear_mb(mb, value, value_fc2_w, value_fc2_b)
        return mb.softmax(x=value, axis=-1)

    return ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_units=compute_unit,
        compute_precision=compute_precision,
    )


def benchmark_predict(np: Any, model: Any, warmup: int, iterations: int) -> dict[str, Any]:
    sample = np.zeros((1, 64, 112), dtype=np.float32)
    for _ in range(warmup):
        model.predict({"x": sample})
    latencies: list[float] = []
    last = None
    for _ in range(iterations):
        start = time.perf_counter()
        last = model.predict({"x": sample})
        latencies.append((time.perf_counter() - start) * 1000.0)
    return {
        "iterations": iterations,
        "median_ms": statistics.median(latencies),
        "mean_ms": statistics.fmean(latencies),
        "p90_ms": percentile(latencies, 0.90),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "output": {k: v.tolist() for k, v in (last or {}).items()},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    np, ct, mb, types = load_coremltools()
    net = load_weights_file(Path(args.weights))
    info = inspect_t1(net)
    if info["encoder_layers"] < 1 or info["ip_emb_channels"] != 256:
        raise RuntimeError("this experimental exporter currently expects the T1-256 attention net")
    compute_unit = compute_unit_from_name(ct, args.compute_unit)
    compute_precision = precision_from_name(ct, args.precision)
    start = time.perf_counter()
    model = build_value_model(np, ct, mb, types, net, compute_unit, compute_precision)
    build_ms = (time.perf_counter() - start) * 1000.0
    package_path = ""
    if args.save_model:
        package_path = str(Path(args.save_model).resolve())
        model.save(package_path)
    result = {
        "weights": str(Path(args.weights).resolve()),
        "compute_unit": args.compute_unit,
        "precision": args.precision,
        "build_ms": build_ms,
        "model_package": package_path,
        "network": info,
    }
    if args.benchmark:
        result["benchmark"] = benchmark_predict(np, model, args.warmup, args.iterations)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental Core ML value-head export for the Lc0 T1-256 net."
    )
    parser.add_argument("weights", help="Lc0 .pb or .pb.gz weights")
    parser.add_argument(
        "--compute-unit",
        choices=["all", "cpu", "cpu-gpu", "cpu-ne"],
        default="cpu-ne",
    )
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--save-model", default="")
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
        print("MetalFish Lc0 Core ML value export")
        print(f"  Weights:      {result['weights']}")
        print(f"  Compute unit: {result['compute_unit']}")
        print(f"  Precision:    {result['precision']}")
        print(f"  Build time:   {result['build_ms']:.1f} ms")
        print(f"  Network:      {result['network']}")
        if result["model_package"]:
            print(f"  Saved model:  {result['model_package']}")
        if "benchmark" in result:
            bench = result["benchmark"]
            print(
                "  Predict:      "
                f"median={bench['median_ms']:.3f} ms "
                f"mean={bench['mean_ms']:.3f} ms p90={bench['p90_ms']:.3f} ms"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
