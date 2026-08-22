"""Parses decklist files (data/*.txt): which cards, how many. Card
definitions live in game.registry.CARD_DEFS; a decklist supplies only
names and quantities.
"""

import re

from . import registry

_LINE_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")


def parse_decklist_text(text):
    """Parses lines shaped "<qty> <name>" into [(name, qty), ...]. Lines
    that don't match (blank lines, headers, comments) are skipped.
    Duplicate lines for the same name are not merged.

    Raises ValueError naming every card with no game.CARD_DEFS entry.
    """
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
    # Explicit UTF-8: decklists carry accented names (Lórien Revealed) and
    # the platform-default encoding (cp1252 on Windows) would mojibake them.
    with open(path, encoding="utf-8") as f:
        return parse_decklist_text(f.read())
