#!/usr/bin/env python3
"""Build a curated engine repertoire book for MetalFish.

Sources opening lines from the lichess-org/chess-openings database (verified PGN),
filters for strong/complex systems, and produces a Polyglot .bin file.

Requires: the lichess-org/chess-openings repo cloned to /tmp/chess-openings
    git clone --depth 1 https://github.com/lichess-org/chess-openings.git /tmp/chess-openings

Usage:
    python3 tools/build_repertoire_book.py [--output books/metalfish_repertoire.bin]
"""

import argparse
import csv
import struct
from pathlib import Path

import chess
import chess.polyglot

OPENINGS_DIR = Path("/tmp/chess-openings")

PREFERRED_WHITE = [
    "Ruy Lopez",
    "Italian Game",
    "Sicilian Defense: Open",
    "Sicilian Defense: Najdorf",
    "Sicilian Defense: Dragon",
    "Sicilian Defense: Sveshnikov",
    "Sicilian Defense: Scheveningen",
    "Sicilian Defense: Classical",
    "Sicilian Defense: Rossolimo",
    "Sicilian Defense: Moscow",
    "Catalan Opening",
    "Queen's Gambit Declined",
    "Queen's Gambit Declined: Exchange",
    "Queen's Gambit Declined: Ragozin",
    "Nimzo-Indian Defense",
    "French Defense: Advance",
    "French Defense: Tarrasch",
    "French Defense: Winawer",
    "Caro-Kann Defense: Advance",
    "Caro-Kann Defense: Classical",
    "Scotch Game",
    "Evans Gambit",
    "English Opening",
    "King's Indian Defense: Classical",
    "Grunfeld Defense",
]

PREFERRED_BLACK = [
    "Sicilian Defense: Najdorf",
    "Sicilian Defense: Sveshnikov",
    "Sicilian Defense: Taimanov",
    "Sicilian Defense: Kan",
    "Sicilian Defense: Scheveningen",
    "Sicilian Defense: Classical",
    "Nimzo-Indian Defense",
    "Queen's Indian Defense",
    "King's Indian Defense",
    "King's Indian Defense: Classical",
    "King's Indian Defense: Mar del Plata",
    "Grunfeld Defense",
    "Grunfeld Defense: Exchange",
    "Queen's Gambit Declined",
    "Queen's Gambit Declined: Ragozin",
    "Semi-Slav Defense",
    "Slav Defense",
    "Ruy Lopez: Marshall Attack",
    "Ruy Lopez: Berlin",
    "Petroff Defense",
    "French Defense: Winawer",
]

AVOID_OPENINGS = [
    "Hippopotamus",
    "Grob",
    "Borg",
    "Sodium Attack",
    "Amar",
    "St. George",
    "Owen",
    "Ware",
    "Polish",
    "Clemenz",
    "Kadas",
    "Mieses",
    "Van Geet",
    "Barnes",
    "Gedult",
    "Goldsmith",
    "Creepy Crawly",
    "Desprez",
    "Durkin",
    "Global",
    "Saragossa",
    "King's Pawn: Wayward",
    "King's Pawn: Busch-Gass",
    "King's Pawn: McConnell",
    "Latvian Gambit",
    "Elephant Gambit",
    "Englund Gambit",
]

HIGH_WEIGHT_ECOS = {
    "B90",
    "B91",
    "B92",
    "B93",
    "B94",
    "B95",
    "B96",
    "B97",
    "B98",
    "B99",
    "B33",
    "B32",
    "C60",
    "C61",
    "C62",
    "C63",
    "C64",
    "C65",
    "C66",
    "C67",
    "C68",
    "C69",
    "C70",
    "C71",
    "C72",
    "C73",
    "C74",
    "C75",
    "C76",
    "C77",
    "C78",
    "C79",
    "C80",
    "C81",
    "C82",
    "C83",
    "C84",
    "C85",
    "C86",
    "C87",
    "C88",
    "C89",
    "C90",
    "C91",
    "C92",
    "C93",
    "C94",
    "C95",
    "C96",
    "C97",
    "C98",
    "C99",
    "E20",
    "E21",
    "E22",
    "E23",
    "E24",
    "E25",
    "E26",
    "E27",
    "E28",
    "E29",
    "E30",
    "E31",
    "E32",
    "E33",
    "E34",
    "E35",
    "E36",
    "E37",
    "E38",
    "E39",
    "E60",
    "E61",
    "E62",
    "E63",
    "E64",
    "E65",
    "E66",
    "E67",
    "E68",
    "E69",
    "E70",
    "E71",
    "E72",
    "E73",
    "E74",
    "E75",
    "E76",
    "E77",
    "E78",
    "E79",
    "E80",
    "E81",
    "E82",
    "E83",
    "E84",
    "E85",
    "E86",
    "E87",
    "E88",
    "E89",
    "E90",
    "E91",
    "E92",
    "E93",
    "E94",
    "E95",
    "E96",
    "E97",
    "E98",
    "E99",
    "D70",
    "D71",
    "D72",
    "D73",
    "D74",
    "D75",
    "D76",
    "D77",
    "D78",
    "D79",
    "D80",
    "D81",
    "D82",
    "D83",
    "D84",
    "D85",
    "D86",
    "D87",
    "D88",
    "D89",
    "E04",
    "E05",
    "E06",
    "E07",
    "E08",
    "E09",
}

MED_WEIGHT_ECOS = {
    "C50",
    "C51",
    "C52",
    "C53",
    "C54",
    "C55",
    "C56",
    "C57",
    "C58",
    "C59",
    "C42",
    "C43",
    "D10",
    "D11",
    "D12",
    "D13",
    "D14",
    "D15",
    "D16",
    "D17",
    "D18",
    "D19",
    "D30",
    "D31",
    "D32",
    "D33",
    "D34",
    "D35",
    "D36",
    "D37",
    "D38",
    "D39",
    "D40",
    "D41",
    "D42",
    "D43",
    "D44",
    "D45",
    "D46",
    "D47",
    "D48",
    "D49",
    "C00",
    "C01",
    "C02",
    "C03",
    "C04",
    "C05",
    "C06",
    "C07",
    "C08",
    "C09",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "B10",
    "B11",
    "B12",
    "B13",
    "B14",
    "B15",
    "B16",
    "B17",
    "B18",
    "B19",
    "B20",
    "B21",
    "B22",
    "B23",
    "B24",
    "B25",
    "B26",
    "B27",
    "B28",
    "B29",
    "B40",
    "B41",
    "B42",
    "B43",
    "B44",
    "B45",
    "B46",
    "B47",
    "B48",
    "B49",
    "B50",
    "B51",
    "B52",
    "B53",
    "B54",
    "B55",
    "B56",
    "B57",
    "B58",
    "B59",
    "B60",
    "B61",
    "B62",
    "B63",
    "B64",
    "B65",
    "B66",
    "B67",
    "B68",
    "B69",
    "B70",
    "B71",
    "B72",
    "B73",
    "B74",
    "B75",
    "B76",
    "B77",
    "B78",
    "B79",
    "B80",
    "B81",
    "B82",
    "B83",
    "B84",
    "B85",
    "B86",
    "B87",
    "B88",
    "B89",
    "C44",
    "C45",
    "C46",
    "C47",
    "C48",
    "C49",
    "A10",
    "A11",
    "A12",
    "A13",
    "A14",
    "A15",
    "A16",
    "A17",
    "A18",
    "A19",
    "A20",
    "A21",
    "A22",
    "A23",
    "A24",
    "A25",
    "A26",
    "A27",
    "A28",
    "A29",
    "A30",
    "A31",
    "A32",
    "A33",
    "A34",
    "A35",
    "A36",
    "A37",
    "A38",
    "A39",
    "E10",
    "E11",
    "E12",
    "E13",
    "E14",
    "E15",
    "E16",
    "E17",
    "E18",
    "E19",
}


def load_openings() -> list[dict]:
    rows = []
    for tsv in sorted(OPENINGS_DIR.glob("*.tsv")):
        with open(tsv) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append(row)
    return rows


def pgn_to_uci_moves(pgn: str) -> list[str] | None:
    board = chess.Board()
    uci_moves = []
    tokens = pgn.replace(".", ". ").split()
    for token in tokens:
        if not token or token[0].isdigit() or token in ("*", "1-0", "0-1", "1/2-1/2"):
            continue
        try:
            move = board.parse_san(token)
            uci_moves.append(move.uci())
            board.push(move)
        except (
            chess.InvalidMoveError,
            chess.IllegalMoveError,
            chess.AmbiguousMoveError,
        ):
            return None
    return uci_moves if uci_moves else None


def score_opening(eco: str, name: str) -> int:
    for avoid in AVOID_OPENINGS:
        if avoid.lower() in name.lower():
            return 0

    if eco in HIGH_WEIGHT_ECOS:
        base = 95
    elif eco in MED_WEIGHT_ECOS:
        base = 80
    else:
        base = 55

    for pref in PREFERRED_WHITE + PREFERRED_BLACK:
        if name.startswith(pref):
            base = max(base, 92)
            break

    return base


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

    raw = (
        (to_file)
        | (to_row << 3)
        | (from_file << 6)
        | (from_row << 9)
        | (promotion << 12)
    )
    return raw


def build_book(output_path: Path, min_ply: int = 4):
    openings = load_openings()
    print(f"Loaded {len(openings)} openings from lichess-org/chess-openings")

    raw_weights: dict[tuple[int, int], int] = {}
    popularity: dict[tuple[int, int], int] = {}
    included = 0
    skipped = 0

    for row in openings:
        eco = row.get("eco", "")
        name = row.get("name", "")
        pgn = row.get("pgn", "")

        weight = score_opening(eco, name)
        if weight == 0:
            skipped += 1
            continue

        uci_moves = pgn_to_uci_moves(pgn)
        if not uci_moves or len(uci_moves) < min_ply:
            skipped += 1
            continue

        board = chess.Board()
        for ply, uci_str in enumerate(uci_moves):
            move = chess.Move.from_uci(uci_str)
            if move not in board.legal_moves:
                break

            key = chess.polyglot.zobrist_hash(board)
            raw_move = encode_polyglot_move(move, board)
            entry_id = (key, raw_move)

            depth_weight = max(1, weight - ply * 2)
            if entry_id in raw_weights:
                raw_weights[entry_id] = max(raw_weights[entry_id], depth_weight)
            else:
                raw_weights[entry_id] = depth_weight
            popularity[entry_id] = popularity.get(entry_id, 0) + 1

            board.push(move)

        included += 1

    entries: dict[tuple[int, int], int] = {}
    for entry_id, base_w in raw_weights.items():
        pop = popularity.get(entry_id, 1)
        entries[entry_id] = min(base_w + min(pop - 1, 10), 127)

    entry_list = [(k, m, w) for (k, m), w in entries.items()]
    entry_list.sort(key=lambda e: e[0])

    with open(output_path, "wb") as f:
        for key, raw_move, weight in entry_list:
            f.write(struct.pack(">QHHI", key, raw_move, weight, 0))

    print(f"Included {included} openings, skipped {skipped}")
    print(f"Built {len(entry_list)} unique position-move entries")
    print(f"Output: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


def verify_book(output_path: Path):
    reader = chess.polyglot.open_reader(str(output_path))
    board = chess.Board()

    print("\nVerification - starting position moves:")
    entries = list(reader.find_all(board))
    for e in sorted(entries, key=lambda x: x.weight, reverse=True)[:8]:
        m = e.move() if callable(e.move) else e.move
        print(f"  {board.san(m)} weight={e.weight}")

    print("\nAfter 1.e4:")
    board.push_san("e4")
    entries = list(reader.find_all(board))
    for e in sorted(entries, key=lambda x: x.weight, reverse=True)[:6]:
        m = e.move() if callable(e.move) else e.move
        print(f"  {board.san(m)} weight={e.weight}")

    print("\nAfter 1.d4:")
    board = chess.Board()
    board.push_san("d4")
    entries = list(reader.find_all(board))
    for e in sorted(entries, key=lambda x: x.weight, reverse=True)[:6]:
        m = e.move() if callable(e.move) else e.move
        print(f"  {board.san(m)} weight={e.weight}")

    print("\nAfter 1.e4 c5 2.Nf3 (Sicilian):")
    board = chess.Board()
    for san in ["e4", "c5", "Nf3"]:
        board.push_san(san)
    entries = list(reader.find_all(board))
    for e in sorted(entries, key=lambda x: x.weight, reverse=True)[:6]:
        m = e.move() if callable(e.move) else e.move
        print(f"  {board.san(m)} weight={e.weight}")

    depth = 0
    board = chess.Board()
    while True:
        entries = list(reader.find_all(board))
        if not entries:
            break
        best = max(entries, key=lambda x: x.weight)
        m = best.move() if callable(best.move) else best.move
        board.push(m)
        depth += 1
    print(f"\nMax depth following best moves: {depth} ply")

    reader.close()


def main():
    parser = argparse.ArgumentParser(description="Build MetalFish repertoire book")
    parser.add_argument(
        "--output",
        type=str,
        default="books/metalfish_repertoire.bin",
        help="Output path",
    )
    parser.add_argument(
        "--min-ply",
        type=int,
        default=4,
        help="Minimum ply depth to include an opening",
    )
    args = parser.parse_args()

    if not OPENINGS_DIR.exists():
        print("ERROR: lichess-org/chess-openings not found at /tmp/chess-openings")
        print(
            "Run: git clone --depth 1 https://github.com/lichess-org/chess-openings.git /tmp/chess-openings"
        )
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    build_book(output, args.min_ply)
    verify_book(output)


if __name__ == "__main__":
    main()
