"""Decklist file parsing: the actual contents of a deck (which cards, how
many) come from a plain text file (data/*.txt), not a hand-written Python
list -- see game.registry.CARD_DEFS for where a name's real definition
(type/cost/effect/extra) lives instead, defined exactly once regardless of
how many decks play it. Swapping a deck's exact contents (reweight/add/
remove a card already implemented somewhere) is purely a text-file edit;
no Python change needed at all.
"""

import re

from . import registry

_LINE_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")


def parse_decklist_text(text):
    """[(name, qty), ...] from lines shaped "<qty> <name>" (data/*.txt's
    own format). Any line not matching that shape -- blank lines, a
    leading "Deck"/"Sideboard" header line, comments -- is silently
    skipped, not an error (monster_tron.txt's own "Deck" header line is
    handled this way, no special-casing needed). Duplicate lines for the
    same name aren't merged here -- build_shuffled_library's own
    accumulate-by-extend loop already sums them correctly, so "4 Forest"
    on two separate lines just becomes 7 Forests, the same as one "7
    Forest" line would.

    Raises ValueError naming every card whose name has no game.CARD_DEFS
    entry -- fail loud, not a silent skip, since that's either a real typo
    or a card that genuinely needs engine work first, and either way
    training silently against a smaller deck than intended would be worse
    than an upfront error."""
    decklist = []
    unknown = []
    for line in text.splitlines():
        match = _LINE_RE.match(line)
        if match is None:
            continue
        qty, name = int(match.group(1)), match.group(2)
        if name not in registry.CARD_DEFS:
            unknown.append(name)
            continue
        decklist.append((name, qty))
    if unknown:
        raise ValueError(f"parse_decklist_text: unknown card name(s), no CARD_DEFS entry: {sorted(set(unknown))}")
    return decklist


def parse_decklist_file(path):
    with open(path) as f:
        return parse_decklist_text(f.read())


if __name__ == "__main__":
    # ponytail self-check: no pytest in this project, mirrors the
    # assert-based demo convention -- run via `python -m game.decklist`
    # from src/.
    assert parse_decklist_text("4 Forest\n2 Swamp\n") == [("Forest", 4), ("Swamp", 2)]
    # A header line, blank lines, and a second line for an already-seen
    # name are all handled without special-casing.
    assert parse_decklist_text("Deck\n4 Forest\n\n3 Forest\n") == [("Forest", 4), ("Forest", 3)]

    try:
        parse_decklist_text("4 Not A Real Card\n2 Swamp\n")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Not A Real Card" in str(e)

    print("decklist.py self-check: OK")
