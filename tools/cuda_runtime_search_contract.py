#!/usr/bin/env python3
"""Shared CUDA-vs-Metal runtime search artifact contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchComparisonSpec:
    key: str
    metal_artifact: str
    metal_input_key: str
    metal_input_label: str
    metalfish_env_var: str
    linux_summary: str
    windows_summary: str

    @property
    def gate_env_var(self) -> str:
        return self.metalfish_env_var.replace("METALFISH_", "GATE_", 1)

    @property
    def metal_manifest_key(self) -> str:
        return self.metal_input_key.removeprefix("metal_")


SEARCH_COMPARISONS: tuple[SearchComparisonSpec, ...] = (
    SearchComparisonSpec(
        key="mcts_bk07",
        metal_artifact="metal-mcts-bk07-search.json",
        metal_input_key="metal_mcts_bk07_search_json",
        metal_input_label="Metal MCTS BK.07 search JSON",
        metalfish_env_var="METALFISH_METAL_MCTS_BK07_SEARCH_JSON",
        linux_summary="metal-cuda-mcts-bk07-search-summary.json",
        windows_summary="metal-windows-cuda-mcts-bk07-search-summary.json",
    ),
    SearchComparisonSpec(
        key="mcts_kiwipete",
        metal_artifact="metal-mcts-kiwipete-search.json",
        metal_input_key="metal_mcts_kiwipete_search_json",
        metal_input_label="Metal MCTS kiwipete search JSON",
        metalfish_env_var="METALFISH_METAL_MCTS_KIWIPETE_SEARCH_JSON",
        linux_summary="metal-cuda-mcts-kiwipete-search-summary.json",
        windows_summary="metal-windows-cuda-mcts-kiwipete-search-summary.json",
    ),
    SearchComparisonSpec(
        key="mcts_after_e4",
        metal_artifact="metal-mcts-after-e4-search.json",
        metal_input_key="metal_mcts_after_e4_search_json",
        metal_input_label="Metal MCTS after-e4 search JSON",
        metalfish_env_var="METALFISH_METAL_MCTS_AFTER_E4_SEARCH_JSON",
        linux_summary="metal-cuda-mcts-after-e4-search-summary.json",
        windows_summary="metal-windows-cuda-mcts-after-e4-search-summary.json",
    ),
    SearchComparisonSpec(
        key="mcts_promotion",
        metal_artifact="metal-mcts-promotion-search.json",
        metal_input_key="metal_mcts_promotion_search_json",
        metal_input_label="Metal MCTS promotion search JSON",
        metalfish_env_var="METALFISH_METAL_MCTS_PROMOTION_SEARCH_JSON",
        linux_summary="metal-cuda-mcts-promotion-search-summary.json",
        windows_summary="metal-windows-cuda-mcts-promotion-search-summary.json",
    ),
    SearchComparisonSpec(
        key="mcts_en_passant",
        metal_artifact="metal-mcts-en-passant-search.json",
        metal_input_key="metal_mcts_en_passant_search_json",
        metal_input_label="Metal MCTS en-passant search JSON",
        metalfish_env_var="METALFISH_METAL_MCTS_EN_PASSANT_SEARCH_JSON",
        linux_summary="metal-cuda-mcts-en-passant-search-summary.json",
        windows_summary="metal-windows-cuda-mcts-en-passant-search-summary.json",
    ),
    SearchComparisonSpec(
        key="hybrid_bk07",
        metal_artifact="metal-hybrid-bk07-search.json",
        metal_input_key="metal_hybrid_bk07_search_json",
        metal_input_label="Metal Hybrid BK.07 search JSON",
        metalfish_env_var="METALFISH_METAL_HYBRID_BK07_SEARCH_JSON",
        linux_summary="metal-cuda-hybrid-bk07-search-summary.json",
        windows_summary="metal-windows-cuda-hybrid-bk07-search-summary.json",
    ),
    SearchComparisonSpec(
        key="hybrid_kiwipete",
        metal_artifact="metal-hybrid-kiwipete-search.json",
        metal_input_key="metal_hybrid_kiwipete_search_json",
        metal_input_label="Metal Hybrid kiwipete search JSON",
        metalfish_env_var="METALFISH_METAL_HYBRID_KIWIPETE_SEARCH_JSON",
        linux_summary="metal-cuda-hybrid-kiwipete-search-summary.json",
        windows_summary="metal-windows-cuda-hybrid-kiwipete-search-summary.json",
    ),
    SearchComparisonSpec(
        key="hybrid_after_e4",
        metal_artifact="metal-hybrid-after-e4-search.json",
        metal_input_key="metal_hybrid_after_e4_search_json",
        metal_input_label="Metal Hybrid after-e4 search JSON",
        metalfish_env_var="METALFISH_METAL_HYBRID_AFTER_E4_SEARCH_JSON",
        linux_summary="metal-cuda-hybrid-after-e4-search-summary.json",
        windows_summary="metal-windows-cuda-hybrid-after-e4-search-summary.json",
    ),
    SearchComparisonSpec(
        key="hybrid_promotion",
        metal_artifact="metal-hybrid-promotion-search.json",
        metal_input_key="metal_hybrid_promotion_search_json",
        metal_input_label="Metal Hybrid promotion search JSON",
        metalfish_env_var="METALFISH_METAL_HYBRID_PROMOTION_SEARCH_JSON",
        linux_summary="metal-cuda-hybrid-promotion-search-summary.json",
        windows_summary="metal-windows-cuda-hybrid-promotion-search-summary.json",
    ),
    SearchComparisonSpec(
        key="hybrid_en_passant",
        metal_artifact="metal-hybrid-en-passant-search.json",
        metal_input_key="metal_hybrid_en_passant_search_json",
        metal_input_label="Metal Hybrid en-passant search JSON",
        metalfish_env_var="METALFISH_METAL_HYBRID_EN_PASSANT_SEARCH_JSON",
        linux_summary="metal-cuda-hybrid-en-passant-search-summary.json",
        windows_summary="metal-windows-cuda-hybrid-en-passant-search-summary.json",
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


def search_summary_artifact_names(*, runtime_kind: str) -> set[str]:
    return {
        path.as_posix()
        for path in search_summary_paths(Path(), runtime_kind=runtime_kind).values()
    }


def metal_artifact_paths(build_dir: Path) -> dict[str, Path]:
    return {spec.key: build_dir / spec.metal_artifact for spec in SEARCH_COMPARISONS}
