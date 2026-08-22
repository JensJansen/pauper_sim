"""Two-player reward + seat-perspective helpers for the DRL policy
(_for_player, _lost), used by rl.rewards / rl.training.train."""

import game


def _lost(state, seat_idx):
    """True once someone has won and it wasn't seat_idx."""
    return state.winner is not None and state.winner != seat_idx


def _for_player(state, player_idx, fn):
    """Runs fn(state) with state.active_idx temporarily set to player_idx,
    restoring it afterward. Lets active-player-proxied logic be reused for
    a non-active player. Only safe between turns, not during a resolution,
    since stack/pending_resolution are shared rather than per-player."""
    original = state.active_idx
    state.active_idx = player_idx
    try:
        return fn(state)
    finally:
        state.active_idx = original
