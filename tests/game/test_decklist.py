"""Tests for game.decklist.parse_decklist_text. Migrated from
game/decklist.py's former `if __name__ == "__main__":` self-check block."""

from game.decklist import parse_decklist_text


def test_parses_basic_lines():
    assert parse_decklist_text("4 Forest\n2 Swamp\n") == [("Forest", 4), ("Swamp", 2)]


def test_ignores_header_and_blank_lines_and_merges_nothing():
    # A header line, blank lines, and a second line for an already-seen
    # name are all handled without special-casing.
    assert parse_decklist_text("Deck\n4 Forest\n\n3 Forest\n") == [("Forest", 4), ("Forest", 3)]


def test_unknown_card_name_raises_value_error():
    try:
        parse_decklist_text("4 Not A Real Card\n2 Swamp\n")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Not A Real Card" in str(e)
