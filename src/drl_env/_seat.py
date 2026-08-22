"""Two-player reward + seat-perspective helpers for the DRL policy
(_for_player, _lost) -- read by rl.rewards / rl.training.train, kept apart from the
action-table engine they don't depend on."""

import game


def _lost(state, seat_idx):
    """True once someone has won and it wasn't seat_idx -- the one thing
    every existing 1-player reward_fn (rl.rewards) can't tell on its own:
    state.turn_won/turn_number don't say WHO won, only that the game
    ended. A win (state.winner == seat_idx) or "nobody yet" (state.winner
    is None, including the still-in-progress case) both fall through
    unchanged to whatever the wrapped reward_fn would already compute --
    only an actual loss needs to be forced to 0 here."""
    return state.winner is not None and state.winner != seat_idx


def _for_player(state, player_idx, fn):
    """Runs fn(state) with state.active_idx temporarily set to player_idx,
    then restores it -- lets existing active-player-proxied logic
    (rl.rewards' reward_fns, game.permanent_power's own aura-enchanting
    search, mana.py's Tron-awareness via state.battlefield, ...) be reused
    for a NON-active player (here: whichever seat's OPPONENT this
    observation is being built for) instead of a second, parallel
    implementation of any of it. Safe even though state.stack/
    pending_resolution are shared, not per-player -- this is only ever
    called between turns (never during a resolution), and every property
    this flip actually affects (hand/battlefield/graveyard/library/
    mana_pool/etc.) is genuinely per-player."""
    original = state.active_idx
    state.active_idx = player_idx
    try:
        return fn(state)
    finally:
        state.active_idx = original
