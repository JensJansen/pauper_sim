"""Casting-adjacent resolutions: which graveyard copy is cast/activated
(Flashback/Escape), "may" choices around a cast (transform, copy,
cast-without-paying), a spell's mode/X/Delve-amount/mana-color choices,
gate-free mana sub-decisions (Saruli-style tap-then-choose-color), "may pay
`cost`" prompts (Spell Pierce/Ward), and Madness's cast-or-graveyard
decision."""

from .. import registry
from ..effects.shared import tap_for_cost
from ._core import begin_resolution, complete_resolution


def begin_choose_cast_copy(state, name, on_complete):
    """WHICH physical copy of `name` in your own graveyard is being cast (or
    having its graveyard ability activated) -- Flashback/Escape/graveyard-
    ability. Real Magic 601.2a: the object being cast is chosen when the
    spell is announced, before costs are paid, so with two same-named cards
    it's a real player choice (e.g. it decides which copy a Rooftop Percher
    trigger still finds legal).

    Mandatory -- no decline. A distinct kind rather than reusing
    choose_graveyard_card, whose decline machinery would be permanently-
    illegal noise here; POINTER-only, adding zero fixed action rows.

    Always the caster's own graveyard. on_complete receives the exact
    chosen CardInstance. Callers only open this when 2+ copies exist; with
    one copy drl_env._actions_cast_altzone._graveyard_instance resolves it
    directly."""
    begin_resolution(state, "choose_cast_copy", on_complete, name=name)


def choose_cast_copy_options(state):
    """The matching graveyard INSTANCES themselves, not names -- tells
    same-named copies apart. No dedup, no sort: rl.decision.action_bridge
    masks/executes by object identity."""
    pending = state.pending_resolution
    return [c for c in state.graveyard if c.name == pending["name"]]


def execute_choose_cast_copy_option(state, card):
    """`card` is the exact chosen graveyard CardInstance; the on_complete
    consumer pays costs and resolves using that exact object."""
    complete_resolution(state, card)


def begin_may_transform(state, permanent, revealed_card=None):
    """A "you may transform this creature" choice (Delver of Secrets). Two
    drl_env actions -- "Transform" / "Don't transform" -- back it.
    revealed_card: the top-of-library card that enables the transform,
    logged as the reveal iff the player transforms."""
    begin_resolution(state, "may_transform", lambda s: None, permanent=permanent, revealed_card=revealed_card)


def execute_may_transform(state, do_transform):
    pending = state.pending_resolution
    permanent = pending["permanent"]
    if do_transform:
        # A transform IS a reveal. Log before the flip.
        revealed = pending.get("revealed_card")
        if revealed is not None:
            state.log_event("reveal", card=revealed, source=(permanent.card_def.name, permanent.slot),
                            from_zone="library", reason="delver")
        # Flip to the back face; front_card_def is kept so the permanent
        # reverts if it later leaves the battlefield.
        spec = registry.EFFECT_REGISTRY.get(permanent.card_def.effect_id, {}).get("transform", {})
        from_name = permanent.card_def.name
        permanent.flags["transformed"] = True
        back = spec.get("card_def")
        if back is not None:
            permanent.flags["front_card_def"] = permanent.card_def
            permanent.card_def = back
        state.log_event("transform", permanent=(from_name, permanent.slot), to_card=permanent.card_def.name,
                        power=spec.get("power"), toughness=spec.get("toughness"))
    complete_resolution(state)


def begin_may_copy(state, on_complete):
    """A "you may copy this spell" choice (Chain Lightning's rider, after
    {R}{R} has already been paid). on_complete(state, do_copy: bool)."""
    begin_resolution(state, "may_copy", on_complete)


def execute_may_copy(state, do_copy):
    complete_resolution(state, do_copy)


def begin_may_cast(state, on_complete):
    """A "you may cast this card [without paying its mana cost]" choice --
    Cascade's may-cast of the hit card. Two drl_env actions -- "Cast (may)"
    / "Decline (may)". The caster's own decision (no active_idx flip).
    on_complete(state, do_cast: bool)."""
    begin_resolution(state, "may_cast", on_complete)


def execute_may_cast(state, do_cast):
    complete_resolution(state, do_cast)


_ANY_COLORS = ("W", "U", "B", "R", "G")  # "any color" -- the five colors, never colorless {C}


def begin_choose_mana_color(state, on_complete):
    """"Add one mana of any color" (Chromatic Star): the activating player picks
    one of the five colors. A mana ability, so it resolves immediately (no
    stack); on_complete(state, color) is what actually adds the mana. Always
    has five legal options, so it never softlocks."""
    begin_resolution(state, "choose_mana_color", on_complete)


def choose_mana_color_options(state):
    return list(_ANY_COLORS)


def execute_choose_mana_color(state, color):
    complete_resolution(state, color)


def begin_choose_cast_mode(state, card_def, modes, on_complete):
    """A modal spell's mode choice (601.2b: chosen as the spell is cast,
    before its total cost is calculated) -- shared "Mode 1".."Mode 5"
    drl_env buttons, capped per-card by len(modes). Covers both cast_modes
    (Winding Way: no X) and x_cast_modes (Nyxborn Hydra: mode chosen first,
    X chosen next via begin_choose_cast_x). modes: tuple of
    (extra_legal_or_None, afford_check) pairs, one per mode in registry
    order -- afford_check(state) -> bool reports whether that mode is
    currently payable. on_complete(state, mode_index) fires with the chosen
    zero-based index."""
    begin_resolution(state, "choose_cast_mode", on_complete, card_def=card_def, modes=modes)


def choose_cast_mode_options(state):
    return list(range(len(state.pending_resolution["modes"])))


def execute_choose_cast_mode_option(state, mode_index):
    complete_resolution(state, mode_index)


def begin_choose_cast_x(state, base_cost, max_x, on_complete):
    """An X-cost spell's X value (601.2b: determined before 601.2f's
    total-cost step) -- shared "X=0".."X=10" drl_env buttons, capped
    per-card/mode by max_x. base_cost is this mode's cost before X's
    generic is added; each shared button computes base_cost+n to check
    affordability. on_complete(state, x) fires with the chosen X."""
    begin_resolution(state, "choose_cast_x", on_complete, base_cost=base_cost, max_x=max_x)


def choose_cast_x_options(state):
    return list(range(state.pending_resolution["max_x"] + 1))


def execute_choose_cast_x_option(state, x):
    complete_resolution(state, x)


def begin_choose_delve_amount(state, card_def, max_n, on_complete):
    """Delve's exile-amount choice (702.66) -- shared "Delve 0".."Delve 6"
    drl_env buttons, capped per-card by max_n. Opens before the exile
    sub-cost (begin_exile_n_from_graveyard) and the reduced-cost payment,
    matching real sequencing. card_def lets each button check both
    graveyard size and the reduced cost's affordability. on_complete(state,
    n) fires with the chosen amount."""
    begin_resolution(state, "choose_delve_amount", on_complete, card_def=card_def, max_n=max_n)


def choose_delve_amount_options(state):
    return list(range(state.pending_resolution["max_n"] + 1))


def execute_choose_delve_amount_option(state, n):
    complete_resolution(state, n)


def begin_mana_subdecision(state, source, target_predicate):
    """Opens a gate-free mana ability's own multi-step choice (currently:
    Saruli Caretaker's "tap another creature, then choose a color") --
    deliberately NOT begin_resolution/state.pending_resolution (a single
    slot that would be clobbered; see state.mana_subdecision's docstring)."""
    state.mana_subdecision = {
        # owner: the seat opening this. A subdecision claims exclusive priority
        # -- see GameState.active_mana_subdecision.
        "stage": "choose_target", "source": source, "target_predicate": target_predicate, "target": None,
        "owner": state.active_idx,
    }
    state.log_event("mana_subdecision_begin", source=(source.card_def.name, source.slot))


def execute_mana_subdecision_target(state, target):
    """Records the chosen tap target, then opens the shared choose_color
    stage with Saruli's completion bound: tap the target, then produce mana
    from the resolved source."""
    from ..mana import activate_mana_source, mana_output  # call-time import -- mana imports resolution

    sub = state.mana_subdecision
    sub["target"] = target
    source = sub["source"]
    state.log_event("mana_subdecision_target", target=(target.card_def.name, target.slot))

    def can_produce(state, color):
        # mana_output raises for an out-of-set color on a "flexible" source.
        try:
            mana_output(source, state, color)
        except ValueError:
            return False
        return True

    def on_choose_color(state, color):
        # tap_for_cost logs the target's own tap event (source's tap is
        # logged separately by activate_mana_source).
        tap_for_cost(state, target)
        activate_mana_source(state, source, color)

    begin_mana_color_choice(state, can_produce, on_choose_color)


def begin_mana_color_choice(state, can_produce, on_choose_color):
    """Opens (or re-opens, mutating in place) the one part of a gate-free
    mana ability's multi-step choice that's identical across every such
    ability: offer a small set of colors, then run a completion once one's
    picked. A mana filter can open straight into this with no prior stage;
    Saruli's choose_target stage feeds into it -- neither needs to know the
    other exists.

    can_produce(state, color) -> bool: is this color currently offerable.
    on_choose_color(state, color) -> None: producing/converting mana and any
    other completion side effect, bound as a closure over whatever context
    the caller needs."""
    sub = state.mana_subdecision
    if sub is None:
        # A filter opens straight into this stage with no prior one, so this is
        # also an OPEN site and must stamp the owner.
        sub = {"owner": state.active_idx}
        state.mana_subdecision = sub
    sub["stage"] = "choose_color"
    sub["can_produce"] = can_produce
    sub["on_choose_color"] = on_choose_color


def execute_mana_subdecision_color(state, color):
    """Closes the sub-decision, then runs whichever completion its caller
    bound (see begin_mana_color_choice)."""
    sub = state.mana_subdecision
    on_choose_color = sub["on_choose_color"]
    state.mana_subdecision = None
    state.log_event("mana_subdecision_complete", color=color)
    on_choose_color(state, color)


def begin_pay_unless(state, payer_idx, cost, on_result):
    """A "player may pay `cost` to decide an outcome" prompt. `cost` is a mana
    cost dict ({"generic": 2} for Spell Pierce / Ward; {"B": 1} for Nihil
    Spellbomb's "you may pay {B}"). `payer_idx` decides whether to pay:
    active_idx is flipped to them and restored once decided. Backed by two
    drl_env actions -- "Pay (unless)" (only when affordable) and "Don't pay
    (unless)"; the mana routes through begin_pay_cost. on_result(state,
    paid: bool) fires with the outcome."""
    original = state.active_idx
    state.active_idx = payer_idx
    begin_resolution(state, "pay_unless", None, cost=cost, on_result=on_result, original_idx=original)


def pay_unless_pay(state):
    """The payer chose to pay: hand off to begin_pay_cost (active_idx stays
    on the payer so they tap their own sources); once paid, restore
    active_idx and report paid=True."""
    from ..mana import begin_pay_cost  # call-time import -- mana imports resolution, avoiding a load cycle

    pending = state.pending_resolution
    cost, on_result, original = pending["cost"], pending["on_result"], pending["original_idx"]
    state.pending_resolution = None

    def _paid(state):
        state.active_idx = original
        on_result(state, True)

    begin_pay_cost(state, cost, on_complete=_paid)


def pay_unless_decline(state):
    pending = state.pending_resolution
    on_result, original = pending["on_result"], pending["original_idx"]
    state.pending_resolution = None
    state.active_idx = original
    on_result(state, False)


def _remove_one_from_exile(state, card_def):
    """First state.exile entry for this exact card_def object. CardDefs are
    interned per name, so identity comparison correctly matches a copy of
    this card."""
    entry = next(e for e in state.exile if e[0] is card_def)
    state.exile.remove(entry)


def begin_madness_decision(state, card_def, on_complete):
    """A qualifying card was just exiled by a discard (see
    execute_discard_option) -- offer "cast it for its madness cost" or
    "let it go to the graveyard." Only ever entered via the trigger-queue
    drain in game/effects/triggers.py, once the discard's enclosing action
    is fully done -- never mid-discard."""
    begin_resolution(state, "madness_decision", on_complete, card_def=card_def)


def madness_decision_options(state):
    return ["cast", "decline"]


def execute_madness_decline(state):
    pending = state.pending_resolution
    card_def = pending["card_def"]
    _remove_one_from_exile(state, card_def)
    state.move_card(card_def, state.graveyard)
    state.log_event("zone_move", card=card_def.name, from_zone="exile", to_zone="graveyard", reason="madness_decline")
    complete_resolution(state)


# "cast" isn't handled here -- paying the madness cost needs
# game.mana.begin_pay_cost, which this module can't import (see the
# module docstring) -- see game.effects.madness_and_plot.execute_madness_cast.
