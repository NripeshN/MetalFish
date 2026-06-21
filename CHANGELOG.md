# Changelog

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
