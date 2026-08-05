"""Pregame mulligan: keep-or-mulligan (London Mulligan) and the resulting
bottom-N-cards placement. Re-exported via game.resolution so
`from ..resolution import X` in the catalogs keeps resolving."""

from ._core import begin_resolution, complete_resolution


def begin_mulligan(state, on_complete):
    """Pregame: this player already has an opening 7-card hand (dealt by
    state.new_multiplayer_game_state's own eager draw(7)) --
    decide keep or mulligan (London Mulligan). Driven by
    game.turn._run_mulligan_gen, once per player, before turn 1 ever starts.

    No hand-contents event of its own: the opening hand and every redraw are
    already logged as library->hand "draw" zone_moves by GameState.draw (the
    single generic draw hook), the mulligan itself by execute_mulligan_take's
    "mulligan_take" zone_move, and the London bottoming by the per-card
    "mulligan_bottom" zone_moves. Together those reconstruct exactly what
    each player saw and did, so a separate "mulligan_hand" event would only
    duplicate the draw events."""
    begin_resolution(state, "mulligan_decision", on_complete)


def mulligan_decision_options(state):
    return ["keep", "mulligan"]


def execute_mulligan_keep(state):
    """Keep the current hand. London Mulligan: put a number of cards equal
    to mulligans already taken this game onto the library bottom, model-
    chosen -- opens a "mulligan_bottom" resolution for exactly that many
    (capped at hand size, in case someone ever mulligans past 7) before
    completing; on_complete only runs once the whole keep (bottoming
    included) is done."""
    on_complete = state.pending_resolution["on_complete"]
    n = min(state.mulligans_taken, len(state.hand))
    if n <= 0:
        complete_resolution(state)
        return
    state.pending_resolution = None
    begin_bottom(state, n, on_complete)


def execute_mulligan_take(state):
    """Take a mulligan: shuffle the current hand back into the library,
    redraw a fresh 7, increment mulligans_taken, then offer the same
    keep-or-mulligan decision again -- London Mulligan allows this as many
    times as the model likes, bounded only by library size like any other
    draw."""
    mulliganed = [c.name for c in state.hand]
    state.library.extend(state.hand)
    state.hand = []
    state.rng.shuffle(state.library)
    state.mulligans_taken += 1
    state.log_event("zone_move", cards=mulliganed, from_zone="hand", to_zone="library", reason="mulligan_take")
    on_complete = state.pending_resolution["on_complete"]
    state.pending_resolution = None
    state.draw(7)
    begin_mulligan(state, on_complete)


def begin_bottom(state, n, on_complete):
    """Put exactly n cards from hand on the library bottom, model-chosen
    one at a time, in the order chosen -- London Mulligan's own "any order"
    (never read back by anything in this engine, so pick order = final
    order, same fungible-by-name picking as begin_discard). Deliberately
    not begin_discard itself -- its Madness routing is discard-specific and
    wrong here."""
    begin_resolution(state, "mulligan_bottom", on_complete, remaining=n)
    if not bottom_options(state):
        complete_resolution(state)


def bottom_options(state):
    pending = state.pending_resolution
    if pending["remaining"] <= 0:
        return []
    return sorted({c.name for c in state.hand})


def execute_bottom_option(state, name):
    pending = state.pending_resolution
    card = next(c for c in state.hand if c.name == name)
    state.hand.remove(card)
    state.library.append(card)
    state.log_event("zone_move", card=name, from_zone="hand", to_zone="library_bottom", reason="mulligan_bottom")
    pending["remaining"] -= 1
    if pending["remaining"] <= 0 or not bottom_options(state):
        complete_resolution(state)
