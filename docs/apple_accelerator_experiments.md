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

## Dense Microbenchmark

`tools/apple_coreml_microbench.py` builds a temporary Core ML ML Program with a
single dense layer and compares Core ML prediction latency against NumPy on the
same shape. It requires `coremltools`; keep that dependency in a temporary venv
unless the branch later gains a real converter pipeline.

Temporary setup used for the first run:

```bash
python3.11 -m venv /tmp/metalfish-coremltools-venv
/tmp/metalfish-coremltools-venv/bin/python -m pip install coremltools==9.0
```

Measured on Apple M2 Max:

| Shape | Compute unit | Core ML median | NumPy median | Result |
| --- | --- | ---: | ---: | --- |
| batch=1, inputs=32, outputs=16 | cpu-ne | 0.0348 ms | 0.0012 ms | CPU wins |
| batch=64, inputs=32, outputs=16 | cpu-ne | 0.0416 ms | 0.0028 ms | CPU wins |
| batch=32, inputs=1024, outputs=256 | cpu-ne | 0.2086 ms | 0.0378 ms | CPU wins |
| batch=32, inputs=1024, outputs=256 | all | 0.2351 ms | 0.0405 ms | CPU wins |

## Decision

Do not move AB NNUE inference to Core ML/ANE based on these measurements. The
per-call Core ML runtime overhead is too high for the small dense operations
that remain after NNUE accumulator updates.

Core ML/ANE may still be worth revisiting for larger batched transformer
subgraphs, but Metal/MPSGraph is already the production transformer path. Any
future ANE work should start with a transformer-shaped benchmark, not NNUE.
