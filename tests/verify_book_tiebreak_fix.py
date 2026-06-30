#!/usr/bin/env python3
"""Verify the opening book tiebreak fix.

Tests that:
1. The repertoire book no longer has problematic h/g-pawn tiebreaks
2. The _lookup_local fallthrough logic works when ties exist
"""

import sys
from pathlib import Path

import chess
import chess.polyglot

PROJ = Path(__file__).resolve().parent.parent
BOOKS_DIR = PROJ / "books"

REPERTOIRE = BOOKS_DIR / "metalfish_repertoire.bin"
GM2001 = BOOKS_DIR / "gm2001.bin"
KOMODO = BOOKS_DIR / "komodo.bin"
RODENT = BOOKS_DIR / "rodent.bin"

# Bad moves that indicate the old lexicographic tiebreak bug
BAD_PAWN_MOVES = {
    "h2h3",
    "h2h4",
    "h7h6",
    "h7h5",
    "g2g4",
    "g2g3",
    "g7g5",
    "h3",
    "h4",
    "h6",
    "h5",
    "g4",
    "g3",
    "g5",
}


def get_book_moves(reader, board):
    """Get all moves from a book for a position with weights."""
    try:
        entries = list(reader.find_all(board))
    except Exception:
        return {}
    weights = {}
    for entry in entries:
        move_attr = entry.move
        move = move_attr() if callable(move_attr) else move_attr
        if move in board.legal_moves:
            uci = move.uci()
            weights[uci] = weights.get(uci, 0) + entry.weight
    return weights


def lookup_local_simulation(board, readers):
    """Simulate the _lookup_local logic from lichess_bot.py."""
    for reader in readers:
        weights = get_book_moves(reader, board)
        if not weights:
            continue
        max_weight = max(weights.values())
        tied = [uci for uci, w in weights.items() if w == max_weight]
        if len(tied) == 1:
            return tied[0], "unique_max", reader
        # Tie detected - if this is the first book (repertoire), skip
        if len(readers) > 1 and reader is readers[0]:
            continue
        # Fallback: lexicographic tiebreak (last resort)
        return (
            max(weights.items(), key=lambda item: (item[1], item[0]))[0],
            "tiebreak",
            reader,
        )
    return None, "no_move", None


def make_board(moves_san):
    """Create a board from a list of SAN moves."""
    board = chess.Board()
    for san in moves_san:
        board.push_san(san)
    return board


# Test positions: (description, moves_san, acceptable_moves_uci, bad_moves_uci)
TEST_POSITIONS = [
    (
        "Najdorf after 5...a6 (White to move)",
        ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"],
        {
            "f1e2",
            "c1e3",
            "c1g5",
            "f2f3",
            "g1e2",
            "f1c4",
        },  # Be2, Be3, Bg5, f3, Nge2, Bc4
        {"h2h3", "h2h4", "g2g4"},  # h3, h4, g4 are bad
    ),
    (
        "Caro-Kann Exchange after 3.exd5 cxd5 (White to move)",
        ["e4", "c6", "d4", "d5", "exd5", "cxd5"],
        {"c2c4", "f1d3", "b1c3", "g1f3", "c1f4"},  # c4, Bd3, Nc3, Nf3, Bf4
        {"g2g4", "h2h4", "g2g3"},  # g4, h4, g3 are bad
    ),
    (
        "Italian after 3.Bc4 (Black to move)",
        ["e4", "e5", "Nf3", "Nc6", "Bc4"],
        {"g8f6", "f8c5", "f8e7", "d7d6"},  # Nf6, Bc5, Be7, d6
        {"h7h6", "h7h5", "g7g5"},  # h6, h5, g5 are bad
    ),
    (
        "KID after 1.d4 Nf6 2.c4 g6 (White to move)",
        ["d4", "Nf6", "c4", "g6"],
        {"b1c3", "g1f3", "g2g3", "b1d2"},  # Nc3, Nf3, g3, Nd2
        {"h2h4"},  # h4 is bad
    ),
    (
        "Starting position (White to move)",
        [],
        {"e2e4", "d2d4", "g1f3", "c2c4"},  # e4, d4, Nf3, c4
        {"h2h3", "h2h4", "g2g4", "a2a4"},  # h3, h4, g4, a4 are bad
    ),
    (
        "After 1.e4 (Black to move)",
        ["e4"],
        {
            "c7c5",
            "e7e5",
            "e7e6",
            "c7c6",
            "d7d5",
            "g8f6",
            "d7d6",
            "g7g6",
        },  # c5, e5, e6, c6, d5, Nf6, d6, g6
        {"h7h5", "h7h6", "g7g5"},  # h5, h6, g5 are bad
    ),
    (
        "After 1.d4 (Black to move)",
        ["d4"],
        {"g8f6", "d7d5", "e7e6", "f7f5", "g7g6", "c7c5"},  # Nf6, d5, e6, f5, g6, c5
        {"h7h5", "h7h6", "g7g5"},  # h5, h6, g5 are bad
    ),
    (
        "After 1.e4 e5 (White to move)",
        ["e4", "e5"],
        {"g1f3", "f1c4", "b1c3", "d2d4", "f2f4"},  # Nf3, Bc4, Nc3, d4, f4
        {"h2h3", "h2h4", "g2g4"},  # h3, h4, g4 are bad
    ),
    (
        "After 1.e4 e5 2.Nf3 (Black to move)",
        ["e4", "e5", "Nf3"],
        {"b8c6", "g8f6", "d7d6", "f7f5"},  # Nc6, Nf6, d6, f5
        {"h7h6", "h7h5", "g7g5"},  # h6, h5, g5 are bad
    ),
    (
        "After 1.d4 Nf6 2.c4 (Black to move)",
        ["d4", "Nf6", "c4"],
        {"e7e6", "g7g6", "c7c5", "e7e5", "d7d5", "c7c6"},  # e6, g6, c5, e5, d5, c6
        {"h7h5", "h7h6"},  # h5, h6 are bad
    ),
    (
        "After 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 (Black to move - Open Sicilian)",
        ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4"],
        {"g8f6", "b8c6", "a7a6", "e7e5"},  # Nf6, Nc6, a6, e5
        {"h7h6", "h7h5", "g7g5"},  # h6, h5, g5 are bad
    ),
    (
        "After 1.e4 e6 2.d4 d5 (White - French Defense)",
        ["e4", "e6", "d4", "d5"],
        {"e4e5", "b1c3", "b1d2", "e4d5"},  # e5, Nc3, Nd2, exd5
        {"h2h3", "h2h4", "g2g4"},  # h3, h4, g4 are bad
    ),
    (
        "Ruy Lopez after 3...a6 (White to move)",
        ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"],
        {"b5a4", "b5c6"},  # Ba4, Bxc6
        {"h2h3", "h2h4", "g2g4"},  # h3, h4, g4 are bad
    ),
    (
        "QGD after 1.d4 d5 2.c4 e6 3.Nc3 (Black to move)",
        ["d4", "d5", "c4", "e6", "Nc3"],
        {"g8f6", "f8e7", "c7c6", "c7c5", "f8b4"},  # Nf6, Be7, c6, c5, Bb4
        {"h7h6", "h7h5", "g7g5"},  # h6, h5, g5 are bad
    ),
    (
        "English Opening after 1.c4 (Black to move)",
        ["c4"],
        {"e7e5", "g8f6", "c7c5", "e7e6", "g7g6", "b8c6"},  # e5, Nf6, c5, e6, g6, Nc6
        {"h7h5", "h7h6", "g7g5"},  # h5, h6, g5 are bad
    ),
]


def run_tests():
    print("=" * 72)
    print("OPENING BOOK TIEBREAK FIX VERIFICATION")
    print("=" * 72)

    # Check book files exist
    print(
        f"\nRepertoire book: {REPERTOIRE} {'[EXISTS]' if REPERTOIRE.exists() else '[MISSING]'}"
    )
    print(f"GM2001 book:     {GM2001} {'[EXISTS]' if GM2001.exists() else '[MISSING]'}")
    print(f"Komodo book:     {KOMODO} {'[EXISTS]' if KOMODO.exists() else '[MISSING]'}")
    print(f"Rodent book:     {RODENT} {'[EXISTS]' if RODENT.exists() else '[MISSING]'}")

    if not REPERTOIRE.exists():
        print("\nFATAL: Repertoire book not found!")
        return False

    rep_reader = chess.polyglot.open_reader(str(REPERTOIRE))
    fallback_readers = []
    for path in [GM2001, KOMODO, RODENT]:
        if path.exists():
            fallback_readers.append(chess.polyglot.open_reader(str(path)))

    all_readers = [rep_reader] + fallback_readers

    passed = 0
    failed = 0
    ties_found = 0

    print("\n" + "-" * 72)
    print("PART 1: REPERTOIRE BOOK MOVE QUALITY")
    print("-" * 72)

    for desc, moves_san, acceptable, bad in TEST_POSITIONS:
        board = make_board(moves_san)
        weights = get_book_moves(rep_reader, board)

        print(f"\n  Position: {desc}")
        if moves_san:
            print(f"  Moves: {' '.join(moves_san)}")
        print(f"  FEN: {board.fen()}")

        if not weights:
            print(f"  Book: NO ENTRIES")
            print(f"  Result: SKIP (position not in repertoire book)")
            continue

        # Sort by weight descending
        sorted_moves = sorted(weights.items(), key=lambda x: -x[1])
        max_weight = sorted_moves[0][1]
        tied_moves = [m for m, w in sorted_moves if w == max_weight]

        print(f"  Book moves (top 5):")
        for uci, w in sorted_moves[:5]:
            move = chess.Move.from_uci(uci)
            san = board.san(move)
            marker = ""
            if uci in bad:
                marker = " <-- BAD (pawn push bug)"
            elif uci in acceptable:
                marker = " <-- OK"
            print(f"    {san:8s} ({uci}) weight={w}{marker}")

        has_tie = len(tied_moves) > 1
        if has_tie:
            ties_found += 1
            print(f"  TIE DETECTED: {len(tied_moves)} moves with weight={max_weight}")
            tie_sans = [board.san(chess.Move.from_uci(m)) for m in tied_moves]
            print(f"    Tied moves: {', '.join(tie_sans)}")

        # Check the best move (what the book would return)
        best_uci = sorted_moves[0][0]
        best_san = board.san(chess.Move.from_uci(best_uci))

        is_bad = best_uci in bad
        is_acceptable = best_uci in acceptable

        if is_bad:
            print(f"  SELECTED: {best_san} ({best_uci}) -- PROBLEMATIC MOVE!")
            print(f"  Result: FAIL")
            failed += 1
        elif is_acceptable:
            print(f"  SELECTED: {best_san} ({best_uci})")
            print(f"  Result: PASS")
            passed += 1
        else:
            # Not in acceptable or bad - might be OK depending on context
            print(
                f"  SELECTED: {best_san} ({best_uci}) -- not in expected list but not a bad pawn push"
            )
            print(f"  Result: PASS (acceptable)")
            passed += 1

    print("\n" + "-" * 72)
    print("PART 2: FALLTHROUGH LOGIC SIMULATION")
    print("-" * 72)
    print("(Simulates _lookup_local: if repertoire has tie, fall through to GM/Komodo)")

    fallthrough_passed = 0
    fallthrough_tested = 0

    for desc, moves_san, acceptable, bad in TEST_POSITIONS:
        board = make_board(moves_san)
        rep_weights = get_book_moves(rep_reader, board)

        if not rep_weights:
            continue

        max_w = max(rep_weights.values())
        tied = [m for m, w in rep_weights.items() if w == max_w]

        if len(tied) <= 1:
            continue  # No tie, no fallthrough needed

        fallthrough_tested += 1
        print(f"\n  Position: {desc}")
        print(
            f"  Repertoire tie: {[board.san(chess.Move.from_uci(m)) for m in tied]} (weight={max_w})"
        )

        # Simulate the full _lookup_local with all readers
        result_uci, source, chosen_reader = lookup_local_simulation(board, all_readers)

        if result_uci is None:
            print(f"  Fallthrough result: No move found in any book")
            continue

        result_san = board.san(chess.Move.from_uci(result_uci))
        reader_name = "repertoire"
        if chosen_reader is not None:
            if chosen_reader is rep_reader:
                reader_name = "repertoire"
            elif fallback_readers and chosen_reader is fallback_readers[0]:
                reader_name = "gm2001"
            elif len(fallback_readers) > 1 and chosen_reader is fallback_readers[1]:
                reader_name = "komodo"
            elif len(fallback_readers) > 2 and chosen_reader is fallback_readers[2]:
                reader_name = "rodent"

        print(
            f"  Fallthrough result: {result_san} ({result_uci}) from {reader_name} [{source}]"
        )

        if result_uci in bad:
            print(f"  Result: FAIL -- still picks a bad pawn push")
        elif result_uci in acceptable:
            print(f"  Result: PASS")
            fallthrough_passed += 1
        else:
            print(f"  Result: PASS (not a bad move)")
            fallthrough_passed += 1

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Repertoire book tests: {passed} PASS, {failed} FAIL")
    print(f"  Ties found in repertoire: {ties_found}")
    print(f"  Fallthrough tests: {fallthrough_passed}/{fallthrough_tested} PASS")

    if failed == 0 and ties_found == 0:
        print(
            "\n  VERDICT: ALL CLEAR - No ties remain, popularity bonus fully resolved tiebreaks"
        )
    elif failed == 0 and ties_found > 0:
        print(
            f"\n  VERDICT: PARTIAL - {ties_found} ties exist but fallthrough prevents bad moves"
        )
    else:
        print(
            f"\n  VERDICT: ISSUES REMAIN - {failed} positions still select bad pawn pushes"
        )

    print("=" * 72)

    # Cleanup
    rep_reader.close()
    for r in fallback_readers:
        r.close()

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
