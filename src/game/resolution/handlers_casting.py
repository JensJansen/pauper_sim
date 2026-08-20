"""Casting-adjacent resolutions: which graveyard copy is being cast/activated
(Flashback/Escape), the small "may" choices around a cast (transform, copy,
cast-without-paying), a spell's own mode/X/Delve-amount/mana-color choices,
gate-free mana sub-decisions (Saruli-style tap-then-choose-color), "may pay
`cost` to decide an outcome" prompts (Spell Pierce/Ward), and Madness's
cast-or-graveyard decision. Re-exported via game.resolution so
`from ..resolution import X` in the catalogs keeps resolving."""

from .. import registry
from ._core import begin_resolution, complete_resolution


def begin_choose_cast_copy(state, name, on_complete):
    """WHICH physical copy of `name` in your own graveyard is being cast (or
    having its graveyard ability activated) -- Flashback/Escape/graveyard-ability.

    Real Magic 601.2a: the object being cast is chosen when the spell is
    ANNOUNCED, before costs are paid, and with two same-named cards in a
    graveyard that is a real choice the PLAYER makes. It is only observable when
    something else references one specific copy -- but then it matters a great
    deal: a Rooftop Percher trigger on the stack targets copy A by object
    identity, and casting copy A in response removes it, making Percher's target
    illegal so that target fizzles (608.2b/c), while casting copy B leaves A to
    be exiled. Both are legal lines; the agent has to be able to aim.

    MANDATORY -- no decline (the caster already committed to casting by taking
    the flashback/activate action, whose own legality already required a
    same-named card here). That is why this is a distinct kind rather than a
    reuse of choose_graveyard_card, whose optional-decline machinery would be
    permanently-illegal noise; it also keeps this a POINTER-only pending that
    adds ZERO fixed action rows (see rl.action_bridge), so no deck's action-space
    width changes.

    This used to also carry a `reserved_cost` -- the cost on_complete was about
    to pay -- stashed so a mana filter taken DURING this choice could be checked
    against it, mana abilities being legal in any priority window (605.1a).
    Removed 2026-08-17: this pending is one of game.mana's mid-cast steps, so no
    mana ability is legal here at all any more (601.2f activates them after the
    copy is chosen, not before), and nothing read the field.

    Always the caster's OWN graveyard (state.graveyard, active-idx proxied) --
    no card in this pool casts from an opponent's graveyard. on_complete receives
    the exact chosen CardInstance. Callers only open this when 2+ copies exist;
    with one copy there is no choice to make (drl_env._actions_cast_altzone._graveyard_instance
    resolves it directly)."""
    begin_resolution(state, "choose_cast_copy", on_complete, name=name)


def choose_cast_copy_options(state):
    """The matching graveyard INSTANCES themselves (objects), not names -- the
    whole point is telling same-named copies apart. No dedup, no sort:
    rl.action_bridge masks/executes by object identity, and the observation
    token for each instance carries that same object (plus a
    targeted_by_mine/targeted_by_theirs bit -- which is what makes "cast the
    copy they're pointing at" a LEARNABLE choice rather than an invisible
    coin flip)."""
    pending = state.pending_resolution
    return [c for c in state.graveyard if c.name == pending["name"]]


def execute_choose_cast_copy_option(state, card):
    """`card` is the exact chosen graveyard CardInstance -- the on_complete
    consumer (a flashback/escape/graveyard-ability execute closure) proceeds to
    pay costs and resolve using that exact object."""
    complete_resolution(state, card)


def begin_may_transform(state, permanent, revealed_card=None):
    """A "you may transform this creature" choice (Delver of Secrets, once an
    instant/sorcery is revealed). Two drl_env actions -- "Transform" /
    "Don't transform" -- back it; execute_may_transform applies or skips.
    revealed_card: the name of the top-of-library card the trigger looked at
    (the instant/sorcery that enables the transform) -- logged as the reveal
    iff the player actually reveals it, i.e. transforms (Delver's "may reveal"
    and "if revealed, transform" collapse to one choice in this engine)."""
    begin_resolution(state, "may_transform", lambda s: None, permanent=permanent, revealed_card=revealed_card)


def execute_may_transform(state, do_transform):
    pending = state.pending_resolution
    permanent = pending["permanent"]
    if do_transform:
        # Revealing the top card is what enables (and, for an I/S, forces) the
        # transform -- so a transform IS a reveal. Log it before the flip.
        revealed = pending.get("revealed_card")
        if revealed is not None:
            state.log_event("reveal", card=revealed, source=(permanent.card_def.name, permanent.slot),
                            from_zone="library", reason="delver")
        # Flip to the back face: set the marker flag (effects.stats reads it for the
        # 3/2-flying stats) AND swap the Permanent's own card_def to the back face, so
        # the game state's identity -- name, every log event, RL perception -- becomes
        # Insectile Aberration, not Delver. front_card_def is kept so the permanent
        # reverts to its front face if it later leaves the battlefield (a DFC is only
        # its back face while on the battlefield).
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
    """A "you may copy this spell" choice (Chain Lightning's rider, after the
    {R}{R} has already been paid -- the second, independent "may" in "they may
    pay {R}{R}. If the player does, they may copy this spell"). on_complete(
    state, do_copy: bool)."""
    begin_resolution(state, "may_copy", on_complete)


def execute_may_copy(state, do_copy):
    complete_resolution(state, do_copy)


def begin_may_cast(state, on_complete):
    """A "you may cast this card [without paying its mana cost]" choice --
    Cascade's own may-cast of the hit card (Maelstrom Colossus). Two drl_env
    actions -- "Cast (may)" / "Decline (may)" -- back it. It is the caster's
    own decision (no active_idx flip). on_complete(state, do_cast: bool)."""
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
    """A modal spell's own mode choice (601.2b: chosen as the spell is cast,
    before its total cost is even calculated) -- shared "Mode 1".."Mode 5"
    drl_env buttons, capped per-card by len(modes), instead of a per-card,
    per-mode fixed-table row. Covers both cast_modes (Winding Way: no X) and
    x_cast_modes (Nyxborn Hydra: mode chosen first, X chosen next via
    begin_choose_cast_x). modes: tuple of (extra_legal_or_None,
    afford_check) pairs, one per mode in this card's own registry order --
    afford_check(state) -> bool reports whether THIS mode is currently
    payable at all (a fixed cost for cast_modes, "some X in range is
    affordable" for x_cast_modes), letting the shared buttons mask per-mode
    without a per-card row. on_complete(state, mode_index) fires with the
    chosen zero-based index into that card's own registry mode order."""
    begin_resolution(state, "choose_cast_mode", on_complete, card_def=card_def, modes=modes)


def choose_cast_mode_options(state):
    return list(range(len(state.pending_resolution["modes"])))


def execute_choose_cast_mode_option(state, mode_index):
    complete_resolution(state, mode_index)


def begin_choose_cast_x(state, base_cost, max_x, on_complete):
    """An X-cost spell's own X value (601.2b: determined before 601.2f's
    total-cost step) -- shared "X=0".."X=10" drl_env buttons, capped per-card/
    mode by max_x, instead of a per-(mode,X) fixed-table row. base_cost is
    this mode's own cost before X's generic is added; each shared button
    computes base_cost+n itself to check affordability -- the exact
    per-value masking a flat per-row enumeration already did, just
    re-encoded as a resolution. on_complete(state, x) fires with the
    chosen X."""
    begin_resolution(state, "choose_cast_x", on_complete, base_cost=base_cost, max_x=max_x)


def choose_cast_x_options(state):
    return list(range(state.pending_resolution["max_x"] + 1))


def execute_choose_cast_x_option(state, x):
    complete_resolution(state, x)


def begin_choose_delve_amount(state, card_def, max_n, on_complete):
    """Delve's own exile-amount choice (702.66) -- shared "Delve 0".."Delve
    6" drl_env buttons, capped per-card by max_n, instead of a per-N
    fixed-table row. Opens BEFORE the existing exile sub-cost
    (begin_exile_n_from_graveyard) and the reduced-cost payment, matching
    real sequencing: choose how much to delve, then pay accordingly.
    card_def lets each shared button check both graveyard size and the
    reduced cost's affordability. on_complete(state, n) fires with the
    chosen amount."""
    begin_resolution(state, "choose_delve_amount", on_complete, card_def=card_def, max_n=max_n)


def choose_delve_amount_options(state):
    return list(range(state.pending_resolution["max_n"] + 1))


def execute_choose_delve_amount_option(state, n):
    complete_resolution(state, n)


def begin_mana_subdecision(state, source, target_predicate):
    """Opens a gate-free mana ability's own multi-step choice (currently:
    Saruli Caretaker's "tap another creature, then choose a color") --
    deliberately NOT begin_resolution/state.pending_resolution (a single
    slot that would be clobbered; see state.mana_subdecision's own
    docstring for the full reasoning). Saruli's own stage-1 entry point --
    "which creature to tap" is a COST CHOICE with no counterpart in a mana
    filter's own flow (drl_env._actions_mana._filter_mana_execute opens straight
    into begin_mana_color_choice below, no target stage at all), so this
    function stays exactly as narrow as it always was; only the SHARED
    choose_color stage (begin_mana_color_choice/execute_mana_subdecision_
    color) needed generalizing to serve both."""
    state.mana_subdecision = {
        # owner: the seat opening this. A subdecision claims EXCLUSIVE priority,
        # and without an owner that lands on whoever is asked next rather than
        # the opener -- see GameState.active_mana_subdecision.
        "stage": "choose_target", "source": source, "target_predicate": target_predicate, "target": None,
        "owner": state.active_idx,
    }
    state.log_event("mana_subdecision_begin", source=(source.card_def.name, source.slot))


def execute_mana_subdecision_target(state, target):
    """Records the chosen tap target, then opens the shared choose_color
    stage with SARULI'S OWN completion behavior bound (tap the target,
    THEN produce mana from the resolved source via a real "mana"-spec
    activation) -- mutates the SAME dict in place (not complete_resolution-
    style clear + fire), since source/target are still needed by
    on_choose_color below and by test/introspection code that reads
    state.mana_subdecision directly."""
    from ..mana import activate_mana_source, mana_output  # call-time import -- mana imports resolution, see pay_unless_pay's own comment

    sub = state.mana_subdecision
    sub["target"] = target
    source = sub["source"]
    state.log_event("mana_subdecision_target", target=(target.card_def.name, target.slot))

    def can_produce(state, color):
        # game.mana_output raises for an out-of-set color on a "flexible"
        # source -- same check _find_mana_source's own color-producibility
        # check already makes for ordinary mana sources. Generic, not
        # hardcoded to Saruli's own 5-true-color case, though Saruli's own
        # ("flexible", set(COLORS)) spec means all 5 always pass here.
        try:
            mana_output(source, state, color)
        except ValueError:
            return False
        return True

    def on_choose_color(state, color):
        # Tap-target-then-activate-source order matches exactly what the
        # pre-generalization implementation did, preserving identical
        # observable behavior.
        target.tapped = True
        activate_mana_source(state, source, color)

    begin_mana_color_choice(state, can_produce, on_choose_color)


def begin_mana_color_choice(state, can_produce, on_choose_color):
    """Opens (Saruli, via execute_mana_subdecision_target above -- or
    re-opens, mutating in place, preserving whatever an earlier stage
    already stashed on the dict) the ONE part of a gate-free mana ability's
    multi-step choice that's genuinely identical across every such ability:
    offer a small set of colors, then run a completion once one's picked.
    Deliberately the ONLY thing this primitive knows about its caller --
    everything upstream (how many stages came before this one, what "cost"
    was already paid to get here) is entirely the caller's own business,
    so a mana filter (drl_env._actions_mana._filter_mana_execute) can open
    straight into this with no prior stage at all, while Saruli's own
    choose_target stage feeds into it, without either needing to know the
    other exists.

    can_produce(state, color) -> bool: is this color currently offerable
    (drl_env._actions_mana._mana_subdecision_color_legal's own check, gated to
    this stage).
    on_choose_color(state, color) -> None: what happens once a color is
    picked (execute_mana_subdecision_color below calls this after clearing
    state.mana_subdecision, mirroring complete_resolution's own "clear
    pending before firing effects" order) -- owns producing/converting mana
    and any other completion side effect (Saruli: tap the stored target,
    then activate the source; a filter: add the chosen color straight to
    the pool). Bound as a closure over whatever context its own caller
    needs (a target, a source, an input color already spent) -- this
    primitive never inspects that context itself."""
    sub = state.mana_subdecision
    if sub is None:
        # A filter opens straight into this stage with no prior one, so this is
        # also an OPEN site and must stamp the owner (see
        # GameState.active_mana_subdecision).
        sub = {"owner": state.active_idx}
        state.mana_subdecision = sub
    sub["stage"] = "choose_color"
    sub["can_produce"] = can_produce
    sub["on_choose_color"] = on_choose_color


def execute_mana_subdecision_color(state, color):
    """Closes the sub-decision, THEN runs whichever completion its caller
    bound (see begin_mana_color_choice) -- clearing state.mana_subdecision
    before firing effects mirrors complete_resolution's own "clear pending
    before on_complete" order."""
    sub = state.mana_subdecision
    on_choose_color = sub["on_choose_color"]
    state.mana_subdecision = None
    state.log_event("mana_subdecision_complete", color=color)
    on_choose_color(state, color)


def begin_pay_unless(state, payer_idx, cost, on_result):
    """A "player may pay `cost` to decide an outcome" prompt. `cost` is a mana
    cost dict ({"generic": 2} for Spell Pierce / Ward; {"B": 1} for Nihil
    Spellbomb's "you may pay {B}"). `payer_idx` (the countered spell's
    controller / the warded permanent's opponent / the Nihil controller)
    decides whether to pay: active_idx is flipped to them for the decision and
    restored once it's made. Backed by two drl_env actions -- "Pay (unless)"
    (only when affordable) and "Don't pay (unless)"; the mana itself routes
    through begin_pay_cost. on_result(state, paid: bool) fires with the
    outcome -- the caller then counters/draws/etc. accordingly."""
    original = state.active_idx
    state.active_idx = payer_idx
    begin_resolution(state, "pay_unless", None, cost=cost, on_result=on_result, original_idx=original)


def pay_unless_pay(state):
    """The payer chose to pay: hand off to begin_pay_cost (active_idx stays on
    the payer so they tap their OWN sources); once the mana is paid, restore
    active_idx and report paid=True."""
    from ..mana import begin_pay_cost  # call-time import -- mana imports resolution, so avoid a load cycle

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
    """First state.exile entry for this exact card_def object -- CardDefs
    are shared/interned per name (game.registry.CARD_DEFS holds one per
    distinct name, not per physical copy), so identity comparison here
    correctly matches "a copy of this card," same fungible-by-name
    convention every other zone in this engine already relies on."""
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
