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
OUTPUT_STAGES = (
    "body",
    "value-embed",
    "value-hidden",
    "value-logits",
    "wdl",
    "policy-embed",
    "policy-qk",
    "policy-raw",
    "policy",
)


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


def compute_precision_for_args(args: argparse.Namespace, ct: Any) -> Any:
    if args.precision == "fp16" and (
        getattr(args, "value_head_fp32", False)
        or getattr(args, "policy_head_fp32", False)
    ):
        from coremltools.converters.mil.mil.passes.defs.quantization import (
            FP16ComputePrecision,
        )

        def op_selector(op: Any) -> bool:
            name = str(getattr(op, "name", ""))
            if getattr(args, "value_head_fp32", False) and name.startswith("value_"):
                return False
            if getattr(args, "policy_head_fp32", False) and name.startswith("policy_"):
                return False
            return True

        return FP16ComputePrecision(op_selector=op_selector)
    return precision_from_name(ct, args.precision)


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


def activation_mb(mb: Any, np: Any, x: Any, activation: str, name: str | None = None) -> Any:
    def named(op: Any, suffix: str, **kwargs: Any) -> Any:
        if name:
            kwargs["name"] = f"{name}_{suffix}"
        return op(**kwargs)

    if activation == "mish":
        one = mb.const(val=np.array(1.0, dtype=np.float32))
        exp = named(mb.exp, "exp", x=x)
        plus = named(mb.add, "plus", x=one, y=exp)
        log = named(mb.log, "log", x=plus)
        tanh = named(mb.tanh, "tanh", x=log)
        return named(mb.mul, "mul", x=x, y=tanh)
    if activation == "swish":
        sigmoid = named(mb.sigmoid, "sigmoid", x=x)
        return named(mb.mul, "mul", x=x, y=sigmoid)
    if activation == "relu":
        return named(mb.relu, "relu", x=x)
    if activation == "relu_2":
        relu = named(mb.relu, "relu", x=x)
        return named(mb.mul, "square", x=relu, y=relu)
    if activation == "selu":
        return named(mb.selu, "selu", x=x)
    raise RuntimeError(f"unsupported activation {activation}")


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


def linear_mb(
    mb: Any, x: Any, weight: Any, bias: Any | None, name: str | None = None
) -> Any:
    kwargs = {
        "x": x,
        "weight": mb.const(val=weight.astype("float32")),
    }
    if bias is not None:
        kwargs["bias"] = mb.const(val=bias.astype("float32"))
    if name:
        kwargs["name"] = name
    return mb.linear(**kwargs)


def slice_by_index_mb(
    mb: Any, np: Any, x: Any, begin: list[int], end: list[int], name: str
) -> Any:
    return mb.slice_by_index(
        x=x,
        begin=np.array(begin, dtype=np.int32),
        end=np.array(end, dtype=np.int32),
        name=name,
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


def parse_attention_policy_map(np: Any) -> Any:
    text = POS_TABLE.read_text(encoding="utf-8")
    marker = "const short kAttnPolicyMap[] = {"
    start = text.index(marker) + len(marker)
    end = text.index("};", start)
    table_text = text[start:end]
    values = [
        int(token)
        for token in table_text.replace("{", " ").replace("}", " ").replace(",", " ").split()
    ]
    if len(values) != 64 * 64 + 8 * 24:
        raise RuntimeError(f"expected 4288 attention policy values, found {len(values)}")
    return np.array(values, dtype=np.int32)


def attention_policy_gather_indices(np: Any) -> Any:
    policy_map = parse_attention_policy_map(np)
    indices = np.full((1858,), -1, dtype=np.int32)
    for scratch_index, policy_index in enumerate(policy_map.tolist()):
        if policy_index >= 0:
            if policy_index >= indices.size:
                raise RuntimeError(f"policy index {policy_index} exceeds 1858 outputs")
            indices[policy_index] = scratch_index
    missing = np.where(indices < 0)[0]
    if missing.size:
        raise RuntimeError(f"attention policy map is missing {missing.size} outputs")
    return indices


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


def activation_name(nf: Any, value: int, default: str) -> str:
    names = {
        1: "mish",
        2: "relu",
        4: "tanh",
        5: "sigmoid",
        6: "selu",
        7: "swish",
        8: "relu_2",
    }
    if value == 0:
        return default
    if value not in names:
        raise RuntimeError(f"unsupported activation enum {value}")
    return names[value]


def effective_activations(net: Any) -> dict[str, str]:
    nf = net.format.network_format
    default = "mish" if nf.default_activation == 1 else "relu"
    smolgen = activation_name(nf, nf.smolgen_activation, default)
    ffn = activation_name(nf, nf.ffn_activation, default)
    if nf.network == 4 and len(net.weights.encoder) > 0 and net.weights.HasField("smolgen_w"):
        smolgen = "swish"
        ffn = "relu_2"
    return {"default": default, "smolgen": smolgen, "ffn": ffn}


def select_policy_head(weights: Any) -> Any:
    if weights.HasField("policy_heads"):
        policy_heads = weights.policy_heads
        for name in ("vanilla", "optimistic_st", "soft", "opponent"):
            if policy_heads.HasField(name):
                return getattr(policy_heads, name)
    return weights


def build_value_model(
    np: Any,
    ct: Any,
    mb: Any,
    types: Any,
    net: Any,
    compute_unit: Any,
    compute_precision: Any,
    output_stage: str = "wdl",
    encoder_layers_limit: int | None = None,
) -> Any:
    weights = net.weights
    activations = effective_activations(net)
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
    policy_head = select_policy_head(weights)

    encoders: list[dict[str, Any]] = []
    for enc in weights.encoder:
        smolgen = None
        if enc.mha.HasField("smolgen"):
            sg = enc.mha.smolgen
            hidden_channels = decode_layer(np, sg.compress).size // channels
            dense1_b = decode_layer(np, sg.dense1_b)
            dense2_b = decode_layer(np, sg.dense2_b)
            smolgen = {
                "compress_w": dense_weight(np, sg.compress, hidden_channels),
                "hidden_channels": hidden_channels,
                "dense1_w": dense_weight(np, sg.dense1_w, dense1_b.size),
                "dense1_b": dense1_b,
                "ln1_g": decode_layer(np, sg.ln1_gammas),
                "ln1_b": decode_layer(np, sg.ln1_betas),
                "dense2_w": dense_weight(np, sg.dense2_w, dense2_b.size),
                "dense2_b": dense2_b,
                "ln2_g": decode_layer(np, sg.ln2_gammas),
                "ln2_b": decode_layer(np, sg.ln2_betas),
            }
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
                "smolgen": smolgen,
            }
        )

    if output_stage not in OUTPUT_STAGES:
        raise RuntimeError(f"unsupported output stage {output_stage}")
    if encoder_layers_limit is not None:
        encoders = encoders[:encoder_layers_limit]

    value_embed_b = decode_layer(np, weights.ip_val_b)
    value_embed_w = dense_weight(np, weights.ip_val_w, value_embed_b.size)
    value_fc1_b = decode_layer(np, weights.ip1_val_b)
    value_fc1_w = dense_weight(np, weights.ip1_val_w, value_fc1_b.size)
    value_fc2_b = decode_layer(np, weights.ip2_val_b)
    value_fc2_w = dense_weight(np, weights.ip2_val_w, value_fc2_b.size)
    policy_embed_b = decode_layer(np, policy_head.ip_pol_b)
    policy_embed_w = dense_weight(np, policy_head.ip_pol_w, policy_embed_b.size)
    policy_q_b = decode_layer(np, policy_head.ip2_pol_b)
    policy_q_w = dense_weight(np, policy_head.ip2_pol_w, policy_q_b.size)
    policy_k_b = decode_layer(np, policy_head.ip3_pol_b)
    policy_k_w = dense_weight(np, policy_head.ip3_pol_w, policy_k_b.size)
    policy_promo_w = dense_weight(np, policy_head.ip4_pol_w, 4)
    policy_d_model = policy_q_b.size
    global_smolgen_w = (
        dense_weight(np, weights.smolgen_w, 64 * 64)
        if weights.HasField("smolgen_w")
        else None
    )
    if policy_d_model == 0 or policy_k_b.size != policy_d_model:
        raise RuntimeError("attention policy q/k weights are missing")
    if len(policy_head.pol_encoder):
        raise RuntimeError("policy encoder layers are not implemented in this experiment")
    policy_indices = attention_policy_gather_indices(np)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, 64, 112), dtype=types.fp32)],
        opset_version=ct.target.iOS17,
    )
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
            if enc["smolgen"] is not None:
                if global_smolgen_w is None:
                    raise RuntimeError("encoder smolgen exists without global smolgen weights")
                sg = enc["smolgen"]
                smolgen = linear_mb(
                    mb, body, sg["compress_w"], None, name="smolgen_compress_linear"
                )
                smolgen = mb.reshape(
                    x=smolgen,
                    shape=[1, 64 * sg["hidden_channels"]],
                    name="smolgen_flatten",
                )
                smolgen = linear_mb(
                    mb, smolgen, sg["dense1_w"], sg["dense1_b"], name="smolgen_dense1"
                )
                smolgen = activation_mb(
                    mb, np, smolgen, activations["smolgen"], name="smolgen_dense1_act"
                )
                smolgen = layer_norm_mb(
                    mb, np, smolgen, None, sg["ln1_g"], sg["ln1_b"], 0.0, 1e-3
                )
                smolgen = linear_mb(
                    mb, smolgen, sg["dense2_w"], sg["dense2_b"], name="smolgen_dense2"
                )
                smolgen = activation_mb(
                    mb, np, smolgen, activations["smolgen"], name="smolgen_dense2_act"
                )
                smolgen = layer_norm_mb(
                    mb, np, smolgen, None, sg["ln2_g"], sg["ln2_b"], 0.0, 1e-3
                )
                smolgen = mb.reshape(
                    x=smolgen,
                    shape=[1, heads, sg["dense2_b"].size // heads],
                    name="smolgen_reshape1",
                )
                smolgen = linear_mb(
                    mb, smolgen, global_smolgen_w, None, name="smolgen_global"
                )
                smolgen = mb.reshape(
                    x=smolgen, shape=[1, heads, 64, 64], name="smolgen_reshape2"
                )
                scores = mb.add(x=scores, y=smolgen, name="smolgen_add")
            probs = mb.softmax(x=scores, axis=-1)
            attn = mb.matmul(x=probs, y=v)
            attn = mb.transpose(x=attn, perm=[0, 2, 1, 3])
            attn = mb.reshape(x=attn, shape=[1, 64, channels])
            attn = linear_mb(mb, attn, enc["dense_w"], enc["dense_b"])
            body = layer_norm_mb(
                mb, np, body, attn, enc["ln1_g"], enc["ln1_b"], alpha, 1e-6
            )

            ffn = linear_mb(mb, body, enc["ffn1_w"], enc["ffn1_b"])
            ffn = activation_mb(mb, np, ffn, activations["ffn"])
            ffn = linear_mb(mb, ffn, enc["ffn2_w"], enc["ffn2_b"])
            body = layer_norm_mb(
                mb, np, body, ffn, enc["ln2_g"], enc["ln2_b"], alpha, 1e-6
            )

        if output_stage == "body":
            return body

        if output_stage.startswith("policy"):
            policy = linear_mb(
                mb, body, policy_embed_w, policy_embed_b, name="policy_embed_linear"
            )
            policy = activation_mb(mb, np, policy, "mish", name="policy_embed_mish")
            if output_stage == "policy-embed":
                return policy

            queries = linear_mb(mb, policy, policy_q_w, policy_q_b, name="policy_q_linear")
            keys = linear_mb(mb, policy, policy_k_w, policy_k_b, name="policy_k_linear")
            keys_t = mb.transpose(x=keys, perm=[0, 2, 1], name="policy_k_transpose")
            policy = mb.matmul(x=queries, y=keys_t, name="policy_qk_matmul")
            scale = mb.const(val=np.array(1.0 / math.sqrt(policy_d_model), dtype=np.float32))
            policy = mb.mul(x=policy, y=scale, name="policy_qk_scale")
            if output_stage == "policy-qk":
                return policy

            promo_keys = slice_by_index_mb(
                mb,
                np,
                keys,
                [0, 56, 0],
                [1, 64, policy_d_model],
                "policy_promo_keys_slice",
            )
            promo_keys = mb.transpose(
                x=promo_keys, perm=[0, 2, 1], name="policy_promo_keys_transpose"
            )
            promo_weights = mb.const(val=policy_promo_w.astype(np.float32))
            promo_logits = mb.matmul(
                x=promo_weights, y=promo_keys, name="policy_promo_weight_matmul"
            )
            offset1 = slice_by_index_mb(
                mb, np, promo_logits, [0, 0, 0], [1, 3, 8], "policy_promo_offset1"
            )
            offset2 = slice_by_index_mb(
                mb, np, promo_logits, [0, 3, 0], [1, 4, 8], "policy_promo_offset2"
            )
            promo = mb.add(x=offset1, y=offset2, name="policy_promo_offset_add")
            promo = mb.stack(
                values=[promo] * 8, axis=3, name="policy_promo_offset_broadcast"
            )
            promo = mb.transpose(
                x=promo, perm=[0, 3, 2, 1], name="policy_promo_offset_transpose"
            )
            promo = mb.reshape(x=promo, shape=[1, 3, 64], name="policy_promo_reshape")

            promo_parent = slice_by_index_mb(
                mb, np, policy, [0, 48, 56], [1, 56, 64], "policy_promo_parent_slice"
            )
            promo_parent = mb.reshape(
                x=promo_parent, shape=[1, 64], name="policy_promo_parent_flatten"
            )
            promo_parent = mb.stack(
                values=[promo_parent] * 3,
                axis=2,
                name="policy_promo_parent_broadcast",
            )
            promo_parent = mb.transpose(
                x=promo_parent, perm=[0, 2, 1], name="policy_promo_parent_transpose"
            )
            promo = mb.add(x=promo, y=promo_parent, name="policy_promo_add")
            policy = mb.concat(values=[policy, promo], axis=1, name="policy_raw_concat")
            if output_stage == "policy-raw":
                return policy

            policy = mb.reshape(x=policy, shape=[1, 4288], name="policy_raw_flatten")
            gather_indices = mb.const(val=policy_indices)
            return mb.gather(
                x=policy,
                indices=gather_indices,
                axis=1,
                validate_indices=False,
                name="policy_map_gather",
            )

        value = linear_mb(mb, body, value_embed_w, value_embed_b, name="value_embed_linear")
        value = activation_mb(mb, np, value, "mish", name="value_embed_mish")
        if output_stage == "value-embed":
            return value
        value = mb.reshape(x=value, shape=[1, 64 * value_embed_b.size], name="value_flatten")
        value = linear_mb(mb, value, value_fc1_w, value_fc1_b, name="value_fc1_linear")
        value = activation_mb(mb, np, value, "mish", name="value_fc1_mish")
        if output_stage == "value-hidden":
            return value
        value = linear_mb(mb, value, value_fc2_w, value_fc2_b, name="value_fc2_linear")
        if output_stage == "value-logits":
            return value
        return mb.softmax(x=value, axis=-1, name="value_wdl")

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
        "output_summary": summarize_prediction_outputs(np, last or {}),
    }


def summarize_prediction_outputs(np: Any, outputs: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, value in outputs.items():
        array = np.asarray(value, dtype=np.float32)
        flat = array.reshape(-1)
        summary[name] = {
            "shape": list(array.shape),
            "min": float(array.min()) if flat.size else 0.0,
            "max": float(array.max()) if flat.size else 0.0,
            "mean": float(array.mean()) if flat.size else 0.0,
            "std": float(array.std()) if flat.size else 0.0,
            "first8": flat[:8].tolist(),
        }
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    np, ct, mb, types = load_coremltools()
    net = load_weights_file(Path(args.weights))
    info = inspect_t1(net)
    if info["encoder_layers"] < 1 or info["ip_emb_channels"] != 256:
        raise RuntimeError("this experimental exporter currently expects the T1-256 attention net")
    compute_unit = compute_unit_from_name(ct, args.compute_unit)
    compute_precision = compute_precision_for_args(args, ct)
    output_stage = getattr(args, "output_stage", "wdl")
    value_head_fp32 = bool(getattr(args, "value_head_fp32", False))
    policy_head_fp32 = bool(getattr(args, "policy_head_fp32", False))
    encoder_layers_arg = getattr(args, "encoder_layers", -1)
    encoder_layers_limit = None if encoder_layers_arg < 0 else encoder_layers_arg
    if encoder_layers_limit is not None and encoder_layers_limit > info["encoder_layers"]:
        raise RuntimeError(
            f"encoder layer probe {encoder_layers_limit} exceeds "
            f"network depth {info['encoder_layers']}"
        )
    start = time.perf_counter()
    model = build_value_model(
        np,
        ct,
        mb,
        types,
        net,
        compute_unit,
        compute_precision,
        output_stage,
        encoder_layers_limit,
    )
    build_ms = (time.perf_counter() - start) * 1000.0
    package_path = ""
    if args.save_model:
        package_path = str(Path(args.save_model).resolve())
        model.save(package_path)
    result = {
        "weights": str(Path(args.weights).resolve()),
        "compute_unit": args.compute_unit,
        "precision": args.precision,
        "value_head_fp32": value_head_fp32,
        "policy_head_fp32": policy_head_fp32,
        "output_stage": output_stage,
        "encoder_layers": encoder_layers_limit,
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
    parser.add_argument(
        "--value-head-fp32",
        action="store_true",
        help="keep the WDL value head in fp32 while using fp16 elsewhere",
    )
    parser.add_argument(
        "--policy-head-fp32",
        action="store_true",
        help="keep the attention policy head in fp32 while using fp16 elsewhere",
    )
    parser.add_argument("--save-model", default="")
    parser.add_argument(
        "--output-stage",
        choices=OUTPUT_STAGES,
        default="wdl",
        help="experimental probe output to return instead of only final WDL",
    )
    parser.add_argument(
        "--encoder-layers",
        type=int,
        default=-1,
        help="limit encoder layers for probe models; -1 uses the full net",
    )
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
        if result["value_head_fp32"]:
            print("  Mixed:       value head fp32")
        if result["policy_head_fp32"]:
            print("  Mixed:       policy head fp32")
        print(f"  Output:       {result['output_stage']}")
        if result["encoder_layers"] is not None:
            print(f"  Encoders:     {result['encoder_layers']}")
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
