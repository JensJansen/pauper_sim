"""Turn loop: untap/draw steps, one full turn, one full game."""

from .effects_common import combat_step, drain_trigger_queue
from .state import new_game_state

MAX_MAIN_PHASE_ACTIONS = 200  # guard against an infinite policy loop, not expected --
# a single "logical" action (cast a spell, activate an ability) can cost
# multiple loop iterations to fully resolve (one per mana tap, plus any
# search/scry/take decisions), where a simpler model might cost exactly one.


def untap_step(state):
    for permanent in state.battlefield:
        permanent.tapped = False
        permanent.summoning_sick = False
        permanent.flags.pop("used_this_turn", None)  # Barrels of Blasting Jelly
    state.mana_pool.clear()  # floating mana doesn't carry across turns


def draw_step(state):
    if state.turn_number == 1 and state.on_the_play:
        return
    state.draw(1)


def run_turn(state, choose_action, combat_enabled=False):
    """One full turn. `choose_action(state)` returns either None ("pass,"
    end the main phase) or a zero-arg callable that performs one complete
    action (mana payment + effect) when invoked.

    combat_enabled: per-deck opt-in (default off, matching every other
    deck-specific knob here) -- only rakdos madness/mono red madness pass
    True. Combat runs once, automatically, at the end of the main phase
    (not the start): a creature already tapped for something else earlier
    in the turn is how a model "holds it back" instead of attacking, no
    separate attack/no-attack decision needed."""
    state.turn_number += 1
    state.lands_played_this_turn = 0
    state.cards_drawn_this_turn = 0
    untap_step(state)
    draw_step(state)
    drain_trigger_queue(state)  # e.g. Sneaky Snacker's return, if the turn's own draw queued it (item 7); no-op otherwise, safe to call unconditionally
    if state.decked_out:
        return  # failed to draw -- loss, same as real Magic's SBA; no main phase this turn

    for _ in range(MAX_MAIN_PHASE_ACTIONS):
        if state.turn_won is not None or state.decked_out:
            break  # already fixed, or a mid-turn draw ability just decked the player out
        action = choose_action(state)
        if action is None:
            break
        action()
        # Drains any queued Madness decision or Sneaky Snacker return --
        # only takes effect once action()'s own resolution is fully done
        # (pending_resolution back to None), never mid-resolution. See
        # docs/MADNESS_DECKS_PLAN.md items 1/3/7.
        drain_trigger_queue(state)

    if combat_enabled and state.turn_won is None and not state.decked_out:
        combat_step(state)


def run_game(decklist, terminated_fn, rng, on_the_play, horizon, choose_action, combat_enabled=False):
    state = new_game_state(decklist, terminated_fn, on_the_play, rng)
    while state.turn_number < horizon and state.turn_won is None and not state.decked_out:
        run_turn(state, choose_action, combat_enabled=combat_enabled)
    return state
