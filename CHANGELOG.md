# Changelog

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
