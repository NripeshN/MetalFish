#!/usr/bin/env python3
"""Shared CUDA-vs-Metal runtime search artifact contract."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchComparisonSpec:
    key: str
    metal_input_key: str
    metal_input_label: str
    linux_summary: str
    windows_summary: str


SEARCH_COMPARISONS: tuple[SearchComparisonSpec, ...] = (
    SearchComparisonSpec(
        key="mcts_bk07",
        metal_input_key="metal_mcts_bk07_search_json",
        metal_input_label="Metal MCTS BK.07 search JSON",
        linux_summary="metal-cuda-mcts-bk07-search-summary.json",
        windows_summary="metal-windows-cuda-mcts-bk07-search-summary.json",
    ),
    SearchComparisonSpec(
        key="mcts_kiwipete",
        metal_input_key="metal_mcts_kiwipete_search_json",
        metal_input_label="Metal MCTS kiwipete search JSON",
        linux_summary="metal-cuda-mcts-kiwipete-search-summary.json",
        windows_summary="metal-windows-cuda-mcts-kiwipete-search-summary.json",
    ),
    SearchComparisonSpec(
        key="hybrid_bk07",
        metal_input_key="metal_hybrid_bk07_search_json",
        metal_input_label="Metal Hybrid BK.07 search JSON",
        linux_summary="metal-cuda-hybrid-bk07-search-summary.json",
        windows_summary="metal-windows-cuda-hybrid-bk07-search-summary.json",
    ),
    SearchComparisonSpec(
        key="hybrid_kiwipete",
        metal_input_key="metal_hybrid_kiwipete_search_json",
        metal_input_label="Metal Hybrid kiwipete search JSON",
        linux_summary="metal-cuda-hybrid-kiwipete-search-summary.json",
        windows_summary="metal-windows-cuda-hybrid-kiwipete-search-summary.json",
    ),
)


def search_comparison_keys() -> tuple[str, ...]:
    return tuple(spec.key for spec in SEARCH_COMPARISONS)


def search_summary_paths(root: Path, *, runtime_kind: str) -> dict[str, Path]:
    if runtime_kind == "linux-cuda":
        return {spec.key: root / spec.linux_summary for spec in SEARCH_COMPARISONS}
    if runtime_kind == "windows-cuda":
        logs = root / "logs"
        return {spec.key: logs / spec.windows_summary for spec in SEARCH_COMPARISONS}
    raise ValueError(f"unsupported CUDA runtime kind: {runtime_kind}")
