# Apple Accelerator Experiments

This branch is for measuring Apple-only backend ideas before they touch search.
The rule is simple: an accelerator path has to beat the current CPU/GPU path in
an isolated benchmark before it is wired into AB, MCTS, or Hybrid.

## Current Probe

`tools/apple_accelerator_probe.py` checks whether the local machine can compile
and run small Swift probes for Core ML, Metal, and MPSGraph.

On the M2 Max test machine:

| Capability | Result |
| --- | --- |
| Core ML framework | available |
| Core ML `cpuAndNeuralEngine` | available |
| Metal device | Apple M2 Max |
| Unified memory | true |
| MPSGraph | available |

This only proves that ANE/Core ML experiments are possible. It does not prove
that they are faster than the existing engine path.

## Microbenchmark Tool

`tools/apple_coreml_microbench.py` builds a temporary Core ML ML Program with a
single dense layer or a transformer-shaped block and compares Core ML prediction
latency against NumPy on the same shape. It requires `coremltools`; keep that
dependency in a temporary venv unless the branch later gains a real converter
pipeline.

Temporary setup used for the first run:

```bash
python3.11 -m venv /tmp/metalfish-coremltools-venv
/tmp/metalfish-coremltools-venv/bin/python -m pip install coremltools==9.0
```

Measured on Apple M2 Max:

Dense-only kernels:

| Shape | Compute unit | Core ML median | NumPy median | Result |
| --- | --- | ---: | ---: | --- |
| batch=1, inputs=32, outputs=16 | cpu-ne | 0.0348 ms | 0.0012 ms | CPU wins |
| batch=64, inputs=32, outputs=16 | cpu-ne | 0.0416 ms | 0.0028 ms | CPU wins |
| batch=32, inputs=1024, outputs=256 | cpu-ne | 0.2086 ms | 0.0378 ms | CPU wins |
| batch=32, inputs=1024, outputs=256 | all | 0.2351 ms | 0.0405 ms | CPU wins |
| batch=32, inputs=1024, outputs=1024 | cpu-ne | 0.2884 ms | 0.1393 ms | CPU wins |
| batch=64, inputs=1024, outputs=1024 | all | 0.3088 ms | 0.1880 ms | CPU wins |

Transformer-shaped single-block kernels:

| Shape | Compute unit | Core ML median | NumPy median | Result |
| --- | --- | ---: | ---: | --- |
| batch=1, tokens=8, channels=32, heads=4, ffn=2x | cpu-ne | 0.0637 ms | 0.0777 ms | Core ML wins narrowly |
| batch=1, tokens=64, channels=128, heads=8, ffn=4x | cpu-ne | 0.3104 ms | 1.0997 ms | Core ML wins |
| batch=8, tokens=64, channels=128, heads=8, ffn=4x | cpu-ne | 0.6771 ms | 7.9926 ms | Core ML wins |
| batch=1, tokens=64, channels=256, heads=8, ffn=4x | cpu-ne | 0.4035 ms | 2.6909 ms | Core ML wins |
| batch=1, tokens=64, channels=512, heads=16, ffn=4x | cpu-ne | 0.5063 ms | 6.1806 ms | Core ML wins |
| batch=1, tokens=64, channels=1024, heads=32, ffn=4x | cpu-ne | 1.4772 ms | 53.1024 ms | Core ML wins |
| batch=1, tokens=64, channels=128, heads=8, ffn=4x | cpu | 1.6400 ms | 1.2066 ms | CPU wins |
| batch=1, tokens=64, channels=128, heads=8, ffn=4x | all | 1.5859 ms | 1.1094 ms | CPU wins |
| batch=1, tokens=64, channels=128, heads=8, ffn=4x | cpu-gpu | 2.0020 ms | 4.2539 ms | Core ML wins, but slower than cpu-ne |

The first Core ML transformer runs used GELU. For a fairer comparison against
the current MPSGraph helper path, `tools/apple_coreml_microbench.py` also
supports `--activation swish`.

Swish transformer-block checks:

| Shape | Backend | Median |
| --- | --- | ---: |
| batch=1, tokens=64, channels=128, heads=8, ffn=4x | Core ML cpu-ne | 0.3133 ms |
| batch=1, tokens=64, channels=128, heads=8, ffn=4x | MPSGraph | 0.6853 ms |
| batch=1, tokens=64, channels=512, heads=16, ffn=4x | Core ML cpu-ne | 0.5704 ms |
| batch=1, tokens=64, channels=512, heads=16, ffn=4x | MPSGraph | 0.8502 ms |
| batch=1, tokens=64, channels=1024, heads=32, ffn=4x | Core ML cpu-ne | 1.4101 ms |
| batch=1, tokens=64, channels=1024, heads=32, ffn=4x | MPSGraph | 1.4827 ms |

Stacked swish transformer checks:

| Shape | Backend | Median |
| --- | --- | ---: |
| layers=2, batch=1, tokens=64, channels=128, heads=8, ffn=4x | Core ML cpu-ne | 0.3051 ms |
| layers=2, batch=1, tokens=64, channels=128, heads=8, ffn=4x | MPSGraph | 0.9004 ms |
| layers=4, batch=1, tokens=64, channels=128, heads=8, ffn=4x | Core ML cpu-ne | 0.4041 ms |
| layers=4, batch=1, tokens=64, channels=128, heads=8, ffn=4x | MPSGraph | 1.2681 ms |
| layers=2, batch=1, tokens=64, channels=512, heads=16, ffn=4x | Core ML cpu-ne | 0.8587 ms |
| layers=2, batch=1, tokens=64, channels=512, heads=16, ffn=4x | MPSGraph | 1.0139 ms |
| layers=4, batch=1, tokens=64, channels=512, heads=16, ffn=4x | Core ML cpu-ne | 1.4058 ms |
| layers=4, batch=1, tokens=64, channels=512, heads=16, ffn=4x | MPSGraph | 1.6838 ms |
| layers=2, batch=1, tokens=64, channels=1024, heads=32, ffn=4x | Core ML cpu-ne | 2.5622 ms |
| layers=2, batch=1, tokens=64, channels=1024, heads=32, ffn=4x | MPSGraph | 1.5283 ms |

Toy whole-network checks include an input embedding, repeated transformer
blocks, a per-token policy projection, and a pooled value projection. These are
still synthetic weights, but they measure more of the inference shape than a
naked transformer block.

| Shape | Backend | Median |
| --- | --- | ---: |
| batch=1, layers=2, tokens=64, input=112, channels=128, heads=8, ffn=4x, policy=32, value=3 | Core ML cpu-ne | 0.3709 ms |
| batch=1, layers=2, tokens=64, input=112, channels=128, heads=8, ffn=4x, policy=32, value=3 | MPSGraph | 1.8700 ms |
| batch=1, layers=2, tokens=64, input=112, channels=512, heads=16, ffn=4x, policy=32, value=3 | Core ML cpu-ne | 0.8872 ms |
| batch=1, layers=2, tokens=64, input=112, channels=512, heads=16, ffn=4x, policy=32, value=3 | MPSGraph | 0.9370 ms |
| batch=1, layers=2, tokens=64, input=112, channels=1024, heads=32, ffn=4x, policy=32, value=3 | Core ML cpu-ne | 1.3202 ms |
| batch=1, layers=2, tokens=64, input=112, channels=1024, heads=32, ffn=4x, policy=32, value=3 | MPSGraph | 1.3323 ms |
| batch=8, layers=2, tokens=64, input=112, channels=128, heads=8, ffn=4x, policy=32, value=3 | Core ML cpu-ne | 0.5468 ms |
| batch=8, layers=2, tokens=64, input=112, channels=128, heads=8, ffn=4x, policy=32, value=3 | MPSGraph | 0.7958 ms |

## Decision

Do not move AB NNUE inference to Core ML/ANE based on these measurements. The
per-call Core ML runtime overhead is too high for the small dense operations
that remain after NNUE accumulator updates.

Core ML/ANE is worth further isolated investigation for transformer-shaped
subgraphs. The `cpu-ne` compute unit is the only promising Core ML mode in the
current data; `cpu`, `all`, and `cpu-gpu` do not justify integration work.

This is still not enough evidence to replace Metal/MPSGraph in the engine. The
single-block and stacked swish data says Core ML `cpu-ne` can beat the current
helper-style MPSGraph block at 128-512 channels, but MPSGraph wins on the
2-layer 1024-channel block shape closest to BT4 width. The toy whole-network
checks narrow that gap, with Core ML winning small shapes and landing at rough
parity for the 1024-channel, 2-layer synthetic network. That is promising for
small side networks, but it is still not a production replacement signal for
BT4: the real network is 15 layers, has real policy/value heads, uses engine
input packing, and depends on batching/caching behavior.

The next useful experiment is narrower: test whether Core ML can accelerate
only small transformer side modules or low-channel experimental networks. Do
not wire Core ML into the production BT4 MCTS path unless a whole-network
BT4-shaped benchmark beats the current Metal/MPSGraph path including input
packing, heads, cache behavior, and batching.
