"""Zone & state model: one simulated game's battlefield/hand/library/graveyard
plus turn/metric bookkeeping.

mana_pool holds mana tapped-but-not-yet-spent (e.g. a tap that produces more
than a cost's live needs) -- spending it, even toward a cost it could cover,
is always its own explicit model action, never automatic. Cleared at the
start of each turn (see turn.untap_step); see game.mana for how it's filled
and spent.
"""

import random

from . import registry


class Permanent:
    """A specific physical card sitting on the battlefield."""

    def __init__(self, card_def, tapped=False):
        self.card_def = card_def
        self.tapped = tapped
        # True until the untap step that first sees this permanent already
        # on the battlefield -- matches real Magic's "under your control
        # since your most recent turn began" (a permanent entering mid-turn,
        # by any path, is sick for the rest of that turn regardless of how
        # it got there). Only combat_step (game.turn) reads this today.
        self.summoning_sick = True
        # Generic per-permanent runtime flags, e.g. "used_mana_filter_this_turn"
        # for Barrels of Blasting Jelly's once-per-turn ability.
        self.flags = {}

    def __repr__(self):
        return f"Permanent({self.card_def.name!r}, tapped={self.tapped})"


class GameState:
    """All zones plus turn/metric bookkeeping for one simulated game."""

    def __init__(self, on_the_play, rng=None, terminated_fn=None):
        self.library = []       # ordered list[CardDef], index 0 = top of deck
        self.hand = []          # list[CardDef]
        self.battlefield = []   # list[Permanent]
        # list[CardDef]. Dread Return (reanimation, Flashback) reads and
        # removes from it, not just bookkeeping.
        self.graveyard = []

        # list[tuple[CardDef, int | None]] -- (card_def, plotted_turn).
        # plotted_turn is None for Madness entries (never outlive one
        # trigger-queue drain, see docs/MADNESS_DECKS_PLAN.md item 3) and
        # state.turn_number-at-the-time for Plot entries (item 4), which
        # persist across turns until cast. Deliberately NOT used by every
        # existing "exiled" card -- Relic of Progenitus and Dread Return's
        # Flashback stay untracked/vanishing, per item 2's decision not to
        # migrate working code nothing ever reads again.
        self.exile = []

        # list[dict], each {"type": "decision"|"automatic", "kind": str, ...}.
        # Populated by things that happen mid-resolution but must not be
        # acted on until the enclosing action's entire effect is fully
        # done (Madness's cast-or-graveyard choice; later, Sneaky
        # Snacker's automatic return -- docs/MADNESS_DECKS_PLAN.md items
        # 1/3/7). Drained by game.effects_common.drain_trigger_queue,
        # called once no other resolution is pending -- never mutated
        # directly by any card's own resolve function.
        self.trigger_queue = []

        self.lands_played_this_turn = 0
        # Reset each turn (turn.run_turn / drl_env._start_turn), incremented
        # once per card actually drawn (see draw() below) -- Sneaky
        # Snacker's "when you draw your third card in a turn" trigger
        # (docs/MADNESS_DECKS_PLAN.md item 7) is the only current reader.
        self.cards_drawn_this_turn = 0
        self.turn_number = 0
        self.on_the_play = on_the_play
        self.rng = rng or random.Random()

        # spy_combo deck: damage accumulated by non-combat damage effects
        # (currently only Lotleth Giant's ETB) against the implicit
        # opponent -- this simulator still has no modeled opponent state
        # beyond this running counter. decked_out flips True the moment
        # draw() is asked for a card with an empty library, which this
        # deck's own combo (mill itself out via Balustrade Spy) can
        # trigger deliberately -- see draw() below.
        self.damage_dealt = 0
        self.decked_out = False

        # terminated_fn(state) -> bool is the per-deck injected win
        # condition, checked generically in enters_battlefield (see
        # game.effects_common) rather than hardcoded to any one deck.
        # Defaults to None so hand-built test states (which set turn_won
        # directly, bypassing this mechanism) don't need to supply one;
        # real games always pass one explicitly via new_game_state.
        self.terminated_fn = terminated_fn
        self.turn_won = None

        # None, or a dict describing an in-progress multi-step decision
        # (paying a cost one tap at a time, resolving a scry/surveil,
        # choosing a search target, ...) that must be fully resolved --
        # via repeated calls into whatever that resolution's own
        # option/execute functions are -- before any other action becomes
        # legal again. See game.resolution.begin_resolution/complete_resolution.
        self.pending_resolution = None

        # dict[str symbol -> int count], e.g. {"G": 2}. Absent/zero entries
        # mean "none floating" -- never holds a "generic" key, only real
        # color/colorless symbols (generic is a cost-side concept, never
        # something a source produces).
        self.mana_pool = {}

    def draw(self, n=1):
        """Real Magic: attempting to draw from an empty library is a loss,
        not a crash. Sets decked_out instead of raising so callers can end
        the episode cleanly -- with turn_won left None, every existing
        reward/scoring function already treats that as an ordinary failure.

        Increments cards_drawn_this_turn per card actually drawn and
        queues an "automatic" trigger-queue entry (drained once the
        enclosing action is fully done, never inline here -- see
        docs/MADNESS_DECKS_PLAN.md items 1/7) for every graveyard card
        whose registry entry has an "on_draw_count" spec matching the new
        count exactly. Generic and registry-driven -- covers Sneaky
        Snacker without this method hardcoding its name; scans the full
        (un-deduped) graveyard, so multiple physical copies each queue
        their own return, matching a per-card triggered ability."""
        for _ in range(n):
            if not self.library:
                self.decked_out = True
                return
            self.hand.append(self.library.pop(0))
            self.cards_drawn_this_turn += 1
            for card_def in self.graveyard:
                spec = registry.EFFECT_REGISTRY.get(card_def.effect_id, {}).get("on_draw_count")
                if spec is not None and self.cards_drawn_this_turn == spec["count"]:
                    self.trigger_queue.append({"type": "automatic", "kind": "on_draw_count", "card_def": card_def})


def build_shuffled_library(decklist, rng):
    """Expand a decklist's quantities into CardDef refs and shuffle. Only
    which decklist's quantities to expand is parameterized -- CARD_DEFS
    stays the single shared name->CardDef lookup (game.registry)."""
    library = []
    for name, qty, *_rest in decklist:
        library.extend([registry.CARD_DEFS[name]] * qty)
    rng.shuffle(library)
    return library


def new_game_state(decklist, terminated_fn, on_the_play, rng):
    state = GameState(on_the_play, rng=rng, terminated_fn=terminated_fn)
    state.library = build_shuffled_library(decklist, rng)
    state.draw(7)
    return state
