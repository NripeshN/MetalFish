#!/usr/bin/env python3
"""Build a curated Polyglot opening book from the Lichess Masters database.

Crawls the lichess masters explorer API, follows only moves with high win rates
and sufficient game counts, and produces a .bin file for use with MetalFish.

Usage:
    python3 tools/build_masters_book.py [--depth 20] [--min-games 50] [--output books/masters.bin]
"""

import argparse
import struct
import time
from pathlib import Path

import chess
import chess.polyglot
import requests

EXPLORER_API = "https://explorer.lichess.ovh/masters"
RATE_LIMIT_DELAY = 1.1


def query_masters(fen: str, session: requests.Session) -> dict | None:
    try:
        r = session.get(
            EXPLORER_API,
            params={"fen": fen, "topGames": 0, "recentGames": 0},
            timeout=10,
        )
        if r.status_code == 429:
            time.sleep(5)
            r = session.get(
                EXPLORER_API,
                params={"fen": fen, "topGames": 0, "recentGames": 0},
                timeout=10,
            )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"  API error for {fen[:40]}...: {e}")
    return None


def compute_weight(move_data: dict, is_white_to_move: bool) -> int:
    white = move_data.get("white", 0)
    draws = move_data.get("draws", 0)
    black = move_data.get("black", 0)
    total = white + draws + black
    if total == 0:
        return 0
    if is_white_to_move:
        win_rate = (white + draws * 0.5) / total
    else:
        win_rate = (black + draws * 0.5) / total
    if win_rate < 0.45:
        return 0
    weight = int(win_rate * 100 * (total**0.3))
    return max(1, min(weight, 65535))


def crawl_position(
    board: chess.Board,
    session: requests.Session,
    entries: list,
    depth: int,
    max_depth: int,
    min_games: int,
    visited: set,
):
    if depth >= max_depth:
        return

    fen = board.fen()
    key = chess.polyglot.zobrist_hash(board)

    if key in visited:
        return
    visited.add(key)

    time.sleep(RATE_LIMIT_DELAY)
    data = query_masters(fen, session)
    if not data:
        return

    moves = data.get("moves", [])
    if not moves:
        return

    is_white = board.turn == chess.WHITE
    candidates = []

    for m in moves:
        total = m.get("white", 0) + m.get("draws", 0) + m.get("black", 0)
        if total < min_games:
            continue
        weight = compute_weight(m, is_white)
        if weight == 0:
            continue
        candidates.append((m, weight, total))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_moves = candidates[:5]

    for move_data, weight, total in top_moves:
        uci_str = move_data["uci"]
        move = chess.Move.from_uci(uci_str)
        if move not in board.legal_moves:
            continue

        raw_move = encode_polyglot_move(move, board)
        entries.append((key, raw_move, weight))

        san = move_data.get("san", uci_str)
        indent = "  " * depth
        print(f"{indent}{san} (games={total}, weight={weight})")

        board.push(move)
        crawl_position(
            board, session, entries, depth + 1, max_depth, min_games, visited
        )
        board.pop()


def encode_polyglot_move(move: chess.Move, board: chess.Board) -> int:
    to_file = chess.square_file(move.to_square)
    to_row = chess.square_rank(move.to_square)
    from_file = chess.square_file(move.from_square)
    from_row = chess.square_rank(move.from_square)

    if board.is_kingside_castling(move):
        to_file = 7
        to_row = from_row
    elif board.is_queenside_castling(move):
        to_file = 0
        to_row = from_row

    promotion = 0
    if move.promotion:
        promotion = {
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
        }.get(move.promotion, 0)

    raw = (to_file) | (to_row << 3) | (from_file << 6) | (from_row << 9) | (promotion << 12)
    return raw


def write_polyglot_bin(entries: list, output_path: Path):
    entries.sort(key=lambda e: e[0])

    with open(output_path, "wb") as f:
        for key, raw_move, weight in entries:
            f.write(struct.pack(">QHHI", key, raw_move, weight, 0))

    print(f"\nWrote {len(entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build masters opening book")
    parser.add_argument("--depth", type=int, default=20, help="Max ply depth")
    parser.add_argument(
        "--min-games", type=int, default=50, help="Minimum games for inclusion"
    )
    parser.add_argument(
        "--output", type=str, default="books/masters.bin", help="Output path"
    )
    parser.add_argument(
        "--token", type=str, default="", help="Lichess API token (or set LICHESS_API_KEY)"
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    import os

    token = args.token or os.environ.get("LICHESS_API_KEY", "")

    session = requests.Session()
    session.headers["Accept"] = "application/json"
    if token:
        session.headers["Authorization"] = f"Bearer {token}"

    board = chess.Board()
    entries = []
    visited = set()

    print(f"Building masters book (depth={args.depth}, min_games={args.min_games})")
    print("Crawling lichess masters database...\n")

    crawl_position(board, session, entries, 0, args.depth, args.min_games, visited)

    if entries:
        write_polyglot_bin(entries, output)
        print(f"Book size: {output.stat().st_size / 1024:.1f} KB")
    else:
        print("No entries collected!")


if __name__ == "__main__":
    main()
