# Hybrid Engine Rewrite: Deep Integration via Shared TT

## Goal
Rewrite the hybrid engine to deeply integrate AB (CPU, ~2M NPS) and MCTS (GPU, ~500 NPS) via shared transposition table in Apple Silicon unified memory. Target: hybrid ≥ max(AB, MCTS) on all benchmarks.

## Architecture
AB-Primary with four bidirectional integration points:
1. MCTS reads AB's TT at leaf nodes (skip NN eval for positions AB already evaluated deeply)
2. Single NN inference at root for AB move ordering (transformer policy → AB root sort)
3. MCTS visit counts → AB root move ordering (published periodically)
4. AB-Primary final decision with conservative MCTS override threshold

## Key Design Decisions
- AB is ALWAYS the primary decision maker (it's 750+ Elo stronger)
- MCTS acts as an advisor, not a competitor
- Zero-copy shared TT leverages unified memory (no PCIe bottleneck)
- MCTS only overrides AB when it has very high confidence AND strong disagreement
- Position classifier determines override thresholds (stricter in tactical positions)

## Testing Strategy
- BK suite: all modes ≥20/24
- paper_benchmarks.py: hybrid ≥ AB score (no regression)
- Each integration point testable independently
