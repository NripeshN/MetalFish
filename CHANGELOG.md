# Changelog

## Unreleased

- New `HybridOversubscribe` option (default off): alpha-beta keeps the full
  `Threads` budget in hybrid mode and the MCTS worker runs as an extra thread.
  Off by default pending game validation.
- Alpha-beta core rebased from Stockfish dev-20260106 (c27c1747) onto
  Stockfish 19 (`sf_19`, edb0d9db): SFNNv16 network with pawn-pair features
  (`nn-1a298aa575a0.nnue`), the small net is gone, and `bench` reproduces the
  upstream signature (2497913 nodes). The MetalFish-specific search hooks
  (MCTS root-order hints, last-PV retention on an unproven fail-high, Apple
  QoS/yield tweaks, TT preservation across thread resizes) are carried over.

## v1.0.0

First stable release.

- NNUE evaluation correctness fix on Apple Silicon: the NEON
  squared-clipped-ReLU now clamps activations at 127, matching the scalar and
  SSE2 paths. This fixes an unsigned-saturation bug that diverged the CPU eval
  from the reference for large activations.
- Hardened UCI option parsing: malformed spin values (e.g. a non-numeric
  `Threads`) are rejected instead of throwing out of the command loop.
- Documentation accuracy pass: honest strength characterization (playing
  strength comes from the alpha-beta + NNUE search; the GPU transformer MCTS is
  a research/diagnostic path, not a strength multiplier), corrected option
  defaults, prebuilt-binary install instructions, and explicit Stockfish/Lc0
  acknowledgements.
- Cross-platform release binaries: tagged releases now publish prebuilt binaries
  for macOS (Apple Silicon), Linux (x86-64 and arm64), and Windows (x86-64), each
  with the NNUE embedded and the BT4 network downloader bundled, plus
  `SHA256SUMS.txt`. (Alpha shipped macOS arm64 only.)
- Release packaging: tagged GitHub releases are marked stable for plain version
  tags and prerelease only for suffixed tags (e.g. `-rc1`).

## v0.1.0-alpha

First public alpha release.

- Apple Silicon ARM release target.
- Hybrid engine promoted as the primary play mode.
- Parallel AB + MCTS search with Metal/MPSGraph BT4 transformer inference.
- Lc0-style MCTS stopping, smart pruning, collision controls, and tree reuse.
- Resource-aware Lichess bot with local opening books, Syzygy, and ponder
  recovery.
- GitHub puzzle runner for long-running Lichess puzzle solving on macOS
  runners.
- Public CI reduced to macOS Metal builds and tests for the alpha release.
