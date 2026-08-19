"""Fold an engine event-stream game log (GameState.log_event format, see
game/state.py) directly into a sequence of board-state snapshots for the
replay viewer -- no intermediate file format, no protobuf, the webapp reads
the raw JSON log itself.

Event-kind interpretation (name-based zone identity tracking via pop-by-name,
DFC face reverts, aura-orphan handling, countered-spell routing, same-phase-
recast/pending-resolution handling -- each one fixing a real bug found
against real logged games, e.g. the Lava Dart flashback identity bug). The
stack is tracked as ONE shared ordered list (top = last), matching the real
GameState.stack -- true LIFO order across both players is exactly the kind
of fidelity this viewer exists for. The pregame mulligan sequence is NOT
netted into one summary step -- see "pregame steps" below.

**Pregame steps (owner directive, 2026-08-02): every mulligan-round draw,
reject, and bottom-card pick is its own step**, not collapsed into one
"opening hands" summary. Each is a genuine, separate decision (its own
mulligan-model forward pass -- see rl/mulligan.py's decide()), which is
exactly the kind of thing this viewer exists to make visible, not hide --
the opposite reasoning from "pass" below, which IS collapsed because it
carries no decision of its own. library_draws (see DEFAULT_DECK_SIZE)
increments on every draw and gives the count back on every mulligan
take/bottom put-back, so it still nets to the right remaining count across
an arbitrary number of mulligan rounds.

Event kinds with no board-visible effect (priority_flip, resolution_begin/
complete, combat_damage/fight_damage -- superseded by life_change, which
fires for the same damage with the total already computed, trigger_fired,
pump/explore/animated -- the log entry doesn't carry enough to render
unambiguously, undercity/initiative/goad/ward status, ...) still produce a
step (so the scrubber timeline never silently skips an event) but don't
mutate board state. A creature's resulting power/toughness from any of
those (a pump, a +1/+1 counter, an attached/orphaned Aura, an animate
threshold) DOES show up, one step later: stats_changed (game.effects.
state_based._log_stat_changes) recomputes every creature's effective stats
before each priority round and logs the delta, which _handle_stats_changed
below applies to the battlefield entry's live power/toughness (kept
alongside its base_power/base_toughness from zone-entry, so the viewer can
tell buffed/debuffed apart from printed).

"pass" is the one exception, owner-authorized (2026-08-02): a priority pass
carries no information the viewer doesn't already show via whatever happens
next -- the stack's top item resolving (its own zone_move step) or the
phase advancing (its own phase_change step) -- so it's dropped rather than
given a step, cutting a large fraction of a typical game's step count that
was pure priority-passing noise.

Phases with nothing but passes in them (both players decline to act, so the
phase does nothing but end) are collapsed too, same day's authorization: a
phase_change is buffered rather than appended immediately, since whether
it's worth showing isn't known until either a real event during that phase
flushes it, or the next phase_change/turn_start discards it unflushed --
an empty phase, skipped straight through to whatever comes next. See
GameReducer._emit.

decision_weights is buffered the same way (owner directive, 2026-08-19,
superseding the 2026-08-02 "always show a real decision, even a pass" call
for this specific case): the network logs one whenever a decision had more
than one legal option, REGARDLESS of whether the chosen action was pass
(rl/agent.py's _log_decision_weights) -- so "the agent considered acting
and passed anyway" was flushing an otherwise-empty phase's header on its
own, defeating the collapse above every time either player merely had an
option to decline. Queued decision_weights steps ride along with the
phase's own buffered header: shown together if a real action follows later
in the same phase, discarded together if it doesn't. A decision that led to
a genuine board change still always shows (the action's own step flushes
both); only pass-only phases disappear.
"""
# The event-stream format never states a deck's true size. Approximate
# "cards remaining": assume a real 60-card constructed deck, decrement once
# per card actually drawn (including
# every mulligan re-draw) and give the credit back on a mulligan/bottom
# put-back, so a multi-mulligan pregame still nets to the right count.
# Owner-authorized simplification (2026-08-02): non-draw library depletion
# (search, mill, impulse exile, scry-to-graveyard) is not counted.
DEFAULT_DECK_SIZE = 60


def pop_by_name(pool, name):
    """Pop and return the first {"name": ...} entry matching name, else None."""
    for i, c in enumerate(pool):
        if c["name"] == name:
            return pool.pop(i)
    return None


def _norm_zone(z):
    """exile_untracked / exile_mesmeric / etc. are all just "exile" to a viewer."""
    return "exile" if z and z.startswith("exile") else z


def _format_candidate_label(c):
    """A decision_weights candidate's display string. rl/agent.py and
    rl/mulligan.py deliberately log structured facts (fixed_label a plain
    string already; pointer_identity a {name, slot, controller} fact),
    never a baked string, so the formatting work happens here -- once, in
    the viewer -- same "log facts, format them in replay_engine.py" pattern
    every other event kind in this file already follows."""
    if c.get("fixed_label") is not None:
        return c["fixed_label"]
    pid = c.get("pointer_identity")
    if not pid:
        return "?"
    parts = [pid.get("name") or "?"]
    if pid.get("slot") is not None:
        parts.append(f"(slot {pid['slot']})")
    if pid.get("controller") is not None:
        parts.append(f"(P{pid['controller']})")
    return " ".join(parts)


class _Player:
    def __init__(self, idx):
        self.idx = idx
        self.life = 20
        self.mana_pool = {}
        self.hand = []               # [{"name": str}]
        self.graveyard = []          # [{"name": str}]
        self.exile = []              # [{"name": str}]
        self.battlefield = {}        # (name, slot) -> entry dict
        self.pending_resolution = []  # [{"name": str}] resolved off stack, fate not yet decided
        self.library_draws = 0       # cards drawn so far this game, see DEFAULT_DECK_SIZE

    def snapshot(self):
        battlefield = sorted(self.battlefield.items(), key=lambda kv: kv[0][1])
        return {
            "life": self.life,
            "library_remaining": max(0, DEFAULT_DECK_SIZE - self.library_draws),
            "mana_pool": dict(self.mana_pool),
            "hand": [c["name"] for c in self.hand],
            "graveyard": [c["name"] for c in self.graveyard],
            "exile": [c["name"] for c in self.exile],
            "battlefield": [
                {
                    "name": entry["name"], "slot": slot, "tapped": entry["tapped"],
                    "power": entry.get("power"), "toughness": entry.get("toughness"),
                    "base_power": entry.get("base_power"), "base_toughness": entry.get("base_toughness"),
                    "is_token": entry.get("is_token", False), "card_type": entry.get("card_type"),
                    "attacking": entry.get("attacking", False), "blocking": entry.get("blocking"),
                    "enchanting": entry.get("enchanting"),
                }
                for (_, slot), entry in battlefield
            ],
        }


class GameReducer:
    def __init__(self, events):
        self.players = [_Player(0), _Player(1)]
        self.stack = []  # [{"name": str, "controller": int}], top = last
        self.steps = []
        self._pending_phase_step = None  # buffered phase_change step, see _emit
        self._pending_decision_steps = []  # buffered decision_weights steps, same phase, see _emit
        # Owner directive (2026-08-02): show every individual mulligan-round
        # draw, reject, and bottom-card pick as its own step -- these are
        # real per-decision moments (each is its own mulligan-model forward
        # pass, see rl/mulligan.py), not bookkeeping to collapse away like
        # "pass". Deliberately NOT netted into one "opening hands" summary
        # (a past design here) -- that hid exactly the keep-vs-mulligan and
        # which-cards-to-bottom decisions this viewer exists to surface.
        self.main_events = events

    def other(self, p):
        return self.players[1 - p.idx]

    # ------------------------------------------------------------------

    def _emit(self, e, kind, description, extra=None):
        """A phase_change is buffered, not appended, since we don't yet know
        whether its phase will contain a real action: the next real event
        flushes it (the phase gets shown after all), while the next
        phase_change/turn_start with nothing having flushed it discards it
        instead -- an empty, priority-passing-only phase, skipped for
        brevity per the owner's request. turn_start is never itself
        buffered (always shown) but also discards a still-pending phase --
        the previous phase ending with nothing in it is exactly the empty
        case this is meant to collapse.

        decision_weights is buffered right alongside it, queued rather than
        appended, for the same reason -- see this module's docstring. extra:
        kind-specific fields merged onto the step (only decision_weights
        uses this -- candidates/chosen_index/value_estimate/network/
        pointer_kind)."""
        step = {
            "kind": kind,
            "turn": e.get("turn"),
            "phase": e.get("phase"),
            "active_player_idx": e.get("active_idx"),
            "turn_player_idx": e.get("turn_player_idx"),
            "description": description,
            "players": [p.snapshot() for p in self.players],
            "stack": [dict(c) for c in self.stack],
        }
        if extra:
            step.update(extra)
        if kind == "phase_change":
            # A new phase starting means the OLD phase is done -- whatever was
            # still buffered from it (header + any pass-only decisions) never
            # got flushed by a real event, so it's discarded together here.
            self._pending_phase_step = step
            self._pending_decision_steps = []
            return
        if kind == "decision_weights":
            self._pending_decision_steps.append(step)
            return
        if kind != "turn_start":
            if self._pending_phase_step is not None:
                self._append_step(self._pending_phase_step)
            for pending in self._pending_decision_steps:
                self._append_step(pending)
        self._pending_phase_step = None
        self._pending_decision_steps = []
        self._append_step(step)

    def _append_step(self, step):
        step["index"] = len(self.steps)
        self.steps.append(step)

    def run(self):
        for e in self.main_events:
            handler = self._HANDLERS.get(e["kind"])
            if handler is not None:
                handler(self, e)
            else:
                idx = e.get("active_idx")
                prefix = f"P{idx}: " if idx is not None else ""
                self._emit(e, e["kind"], prefix + e["kind"].replace("_", " "))
        self._flush_pending_resolutions()
        return self.steps

    # ------------------------------------------------------------------
    def _flush_pending_resolutions(self):
        for p in self.players:
            for c in p.pending_resolution:
                p.graveyard.append(c)
            p.pending_resolution = []

    def _clear_combat_flags(self):
        for p in self.players:
            for entry in p.battlefield.values():
                entry["attacking"] = False
                entry["blocking"] = None

    def _find_perm(self, name_slot):
        key = tuple(name_slot)
        for p in self.players:
            if key in p.battlefield:
                return p, key
        return None, None

    def _pop_stack(self, name):
        for i, c in enumerate(self.stack):
            if c["name"] == name:
                return self.stack.pop(i)
        return None

    # ------------------------------------------------------------------ turn/phase
    def _handle_turn_start(self, e):
        self._flush_pending_resolutions()
        self._emit(e, "turn_start", f"Turn {e.get('turn')}: player {e['turn_player_idx']} begins their turn")

    def _handle_phase_change(self, e):
        if e.get("phase") != "declare_blockers":
            self._clear_combat_flags()
        self._flush_pending_resolutions()
        self._emit(e, "phase_change", f"— {(e.get('phase') or '').replace('_', ' ').title()} —")

    def _handle_pass(self, e):
        """No step: whatever a pass leads to (a stack item resolving, a phase
        advancing) already gets its own step. See module docstring."""

    # ------------------------------------------------------------------ mana
    def _handle_mana_tap(self, e):
        p = self.players[e["active_idx"]]
        name, slot = e["permanent"]
        entry = p.battlefield.get((name, slot))
        if entry is not None:
            entry["tapped"] = True
        produced = e.get("produced") or []
        for color in produced:
            p.mana_pool[color] = p.mana_pool.get(color, 0) + 1
        self._emit(e, "mana_tap", f"P{p.idx} taps {name} for {','.join(produced) or '?'}")

    def _handle_mana_spend(self, e):
        p = self.players[e["active_idx"]]
        color = e["color"]
        p.mana_pool[color] = max(0, p.mana_pool.get(color, 0) - 1)
        self._emit(e, "mana_spend", f"P{p.idx} spends {color} mana")

    def _handle_untap_step(self, e):
        p = self.players[e["active_idx"]]
        for name, slot in e.get("untapped") or []:
            entry = p.battlefield.get((name, slot))
            if entry is not None:
                entry["tapped"] = False
        self._emit(e, "untap_step", f"P{p.idx} untaps")

    def _handle_mana_emptied(self, e):
        for idx_str, pool in (e.get("pools") or {}).items():
            p = self.players[int(idx_str)]
            for color in pool:
                p.mana_pool[color] = 0
        self._emit(e, "mana_emptied", "Mana pools empty (rule 500.4)")

    # ------------------------------------------------------------------ combat
    def _handle_attack_declared(self, e):
        p = self.players[e["active_idx"]]
        name, slot = e["attacker"]
        entry = p.battlefield.get((name, slot))
        if entry is not None:
            entry["attacking"] = True
            if e.get("tapped"):
                entry["tapped"] = True
        self._emit(e, "attack_declared", f"P{p.idx} attacks with {name}")

    def _handle_block_assigned(self, e):
        blocker_p = self.players[e["active_idx"]]
        b_entry = blocker_p.battlefield.get(tuple(e["blocker"]))
        if b_entry is not None:
            b_entry["blocking"] = list(e["attacker"])
        self._emit(e, "block_assigned", f"P{blocker_p.idx} blocks {e['attacker'][0]} with {e['blocker'][0]}")

    # ------------------------------------------------------------------ auras
    def _handle_aura_attached(self, e):
        aura_p = self.players[e["active_idx"]]
        aura = aura_p.battlefield.get(tuple(e["aura"]))
        if aura is not None:
            target_p, target_key = self._find_perm(e["target"])
            if target_key is not None:
                # Resolved to the target's controller here (not just name/slot,
                # which can collide across the two players' battlefields) so the
                # viewer can nest the aura under the right permanent even when
                # it enchants an opponent's (e.g. Pacifism-style).
                aura["enchanting"] = {"controller_idx": target_p.idx, "name": target_key[0], "slot": target_key[1]}
        self._emit(e, "aura_attached", f"P{aura_p.idx} attaches {e['aura'][0]} to {e['target'][0]}")

    def _handle_aura_orphaned(self, e):
        p, key = self._find_perm(e["aura"])
        desc = f"{e['aura'][0]} falls off (its enchanted permanent left)"
        if key is not None:
            old = p.battlefield.pop(key)
            if not old.get("is_token"):
                outcome = e.get("outcome", "graveyard")
                pool = {"hand": p.hand, "exile": p.exile}.get(outcome, p.graveyard)
                pool.append({"name": key[0]})
            desc += f" → {e.get('outcome', 'graveyard')}"
        self._emit(e, "aura_orphaned", desc)

    # ------------------------------------------------------------------ life / death
    def _handle_life_change(self, e):
        p = self.players[e["player_idx"]]
        p.life = e["new_total"]
        amt = e.get("amount")
        sign = f"{amt:+d} " if isinstance(amt, int) else ""
        self._emit(e, "life_change", f"P{p.idx} life {sign}→ {p.life}")

    def _handle_state_based_death(self, e):
        owner = self.players[e["owner_idx"]]
        name, slot = e["permanent"]
        old = owner.battlefield.pop((name, slot), None)
        if old is not None and not old.get("is_token"):
            owner.graveyard.append({"name": old.get("front_name") or name})
        self._emit(e, "state_based_death", f"{name} dies (state-based action)")

    def _handle_destroy(self, e):
        p = self.players[e.get("owner_idx", e["active_idx"])]
        name, slot = e["permanent"]
        old = p.battlefield.pop((name, slot), None)
        if old is not None and not old.get("is_token"):
            to_zone = _norm_zone(e.get("to_zone")) or "graveyard"
            pool = {"graveyard": p.graveyard, "exile": p.exile, "hand": p.hand}.get(to_zone, p.graveyard)
            pool.append({"name": old.get("front_name") or name})
        self._emit(e, "destroy", f"{name} destroyed")

    def _handle_countered(self, e):
        p = self.players[e.get("controller", e["active_idx"])]
        c = self._pop_stack(e["card"])
        if c is not None:
            p.graveyard.append({"name": e["card"]})
        self._emit(e, "countered", f"{e['card']} is countered")

    # ------------------------------------------------------------------ misc named events
    def _handle_put_on_top(self, e):
        p = self.players[e["active_idx"]]
        names = e.get("cards") or []
        for name in names:
            pop_by_name(p.hand, name)
        self._emit(e, "put_on_top", f"P{p.idx} puts {', '.join(names) or '?'} on top of their library")

    def _handle_mill(self, e):
        p = self.players[e["player_idx"]]
        names = e.get("cards") or []
        for name in names:
            p.graveyard.append({"name": name})
        self._emit(e, "mill", f"P{p.idx} mills {', '.join(names) or '?'}")

    def _handle_graveyard_exiled(self, e):
        if "exiled" in e:
            p = self.players[e["target_player_idx"]]
            for name in e["exiled"]:
                c = pop_by_name(p.graveyard, name)
                if c is not None:
                    p.exile.append(c)
            desc = f"P{p.idx} graveyard cards exiled: {', '.join(e['exiled'])}"
        else:
            p = self.players[e["player_idx"]]
            p.exile.extend(p.graveyard)
            p.graveyard = []
            desc = f"P{p.idx} graveyard exiled"
        self._emit(e, "graveyard_exiled", desc)

    def _handle_graveyards_exiled(self, e):
        for p in self.players:
            p.exile.extend(p.graveyard)
            p.graveyard = []
        self._emit(e, "graveyards_exiled", "Both graveyards exiled")

    def _handle_impulse_exile(self, e):
        p = self.players[e["active_idx"]]
        names = e.get("cards") or []
        for name in names:
            p.exile.append({"name": name})
        self._emit(e, "impulse_exile", f"P{p.idx} exiles {', '.join(names) or '?'} (impulse draw)")

    def _set_tapped(self, e, tapped):
        p, key = self._find_perm(e["permanent"])
        if key is not None:
            p.battlefield[key]["tapped"] = tapped
        return key

    def _handle_tap(self, e):
        key = self._set_tapped(e, True)
        self._emit(e, "tap", f"{(key or e['permanent'])[0]} taps")

    def _handle_untap(self, e):
        key = self._set_tapped(e, False)
        self._emit(e, "untap", f"{(key or e['permanent'])[0]} untaps")

    def _handle_tap_or_untap(self, e):
        tapped = bool(e.get("now_tapped"))
        self._set_tapped(e, tapped)
        self._emit(e, "tap_or_untap", f"{e['permanent'][0]} {'taps' if tapped else 'untaps'}")

    def _handle_tuck(self, e):
        p = self.players[e.get("owner_idx", e["active_idx"])]
        key = next((k for k in p.battlefield if k[0] == e["card"]), None)
        if key is not None:
            p.battlefield.pop(key)
        self._emit(e, "tuck", f"{e['card']} put into its owner's library")

    def _handle_transform(self, e):
        from_name, slot = e["permanent"]
        to_card = e.get("to_card")
        p, key = self._find_perm([from_name, slot])
        if key is not None and to_card:
            entry = p.battlefield.pop(key)
            entry["front_name"] = from_name
            entry["name"] = to_card
            if e.get("power") is not None:
                entry["power"] = e.get("power")
                entry["toughness"] = e.get("toughness")
                entry["base_power"] = e.get("power")  # new face's own printed stats
                entry["base_toughness"] = e.get("toughness")
            p.battlefield[(to_card, slot)] = entry
        self._emit(e, "transform", f"{from_name} transforms into {to_card or '?'}")

    def _handle_stats_changed(self, e):
        """game.effects.state_based._log_stat_changes: a creature's effective
        power/toughness (counters, until-EOT pump, Auras, animate/transform,
        static-self boosts -- see that function's docstring) moved off
        whatever was last shown. Updates the LIVE power/toughness only --
        base_power/base_toughness (set at zone-entry/transform) stay put, so
        the viewer can tell buffed/debuffed apart from printed."""
        p, key = self._find_perm(e["permanent"])
        name = key[0] if key is not None else e["permanent"][0]
        if key is not None:
            entry = p.battlefield[key]
            entry["power"] = e.get("power")
            entry["toughness"] = e.get("toughness")
        self._emit(e, "stats_changed", f"{name} is now {e.get('power')}/{e.get('toughness')}")

    def _handle_reveal(self, e):
        p = self.players[e["active_idx"]]
        self._emit(e, "reveal", f"P{p.idx} reveals {e.get('card') or '?'} off the top of their library")

    def _handle_decision_weights(self, e):
        """Non-board-mutating (like the unhandled-kind fallback): carries the
        agent's top-5 candidate actions through to the step dict for the
        viewer's decision panel, formatting each candidate's label here (see
        _format_candidate_label). See todo/game_visualization.md's
        "Decision-point overlay" section."""
        candidates = [
            {"index": c.get("index"), "probability": c.get("probability"), "label": _format_candidate_label(c)}
            for c in (e.get("candidates") or [])
        ]
        network = e.get("network")
        desc = f"Decision weights ({network}): {len(candidates)} candidate(s)"
        self._emit(e, "decision_weights", desc, extra={
            "network": network,
            "chosen_index": e.get("chosen_index"),
            "value_estimate": e.get("value_estimate"),
            "candidates": candidates,
            "pointer_kind": e.get("pointer_kind"),
        })

    # ------------------------------------------------------------------ zone_move (the big dispatcher)
    def _handle_zone_move(self, e):
        if "disposed" in e:
            desc = self._mutate_disposed(e)
        elif "permanent" in e:
            desc = self._mutate_permanent_move(e)
        else:
            name = e.get("card")
            names = [name] if name is not None else (e.get("cards") or [])
            for n in names:
                self._mutate_card_move(e, n)
            frm, to = _norm_zone(e.get("from_zone")), _norm_zone(e.get("to_zone"))
            reason = f" ({e['reason']})" if e.get("reason") else ""
            idx = e.get("active_idx")
            desc = f"P{idx}: {', '.join(names) or '(nothing)'} {frm or '?'} → {to or '?'}{reason}"
        self._emit(e, "zone_move", desc)

    def _mutate_disposed(self, e):
        """Scry/surveil. disposed_to='library_bottom' has no visible zone change
        (deck order isn't tracked); only the graveyard case is a real move."""
        disposed_to = e.get("disposed_to")
        names = e.get("disposed") or []
        p = self.players[e["active_idx"]]
        if disposed_to == "graveyard":
            for name in names:
                p.graveyard.append({"name": name})
        kept = e.get("kept_to_library_top") or []
        parts = []
        if names:
            parts.append(f"{', '.join(names)} → {disposed_to or 'library'}")
        if kept:
            parts.append(f"{', '.join(kept)} kept on top")
        return f"P{p.idx} scry/surveil: " + (", ".join(parts) if parts else "no changes")

    def _resolve_incoming_permanent(self, p, name, from_zone):
        """Remove `name` from wherever it's plausibly entering the battlefield
        from, and report whether this is a token (no prior identity anywhere)."""
        for i, c in enumerate(p.pending_resolution):
            if c["name"] == name:
                p.pending_resolution.pop(i)
                return False
        if from_zone == "hand":
            pop_by_name(p.hand, name)
            return False
        pool = {"graveyard": p.graveyard, "exile": p.exile}.get(from_zone)
        if pool is not None:
            pop_by_name(pool, name)
            return False
        return from_zone is None  # None with nothing pending: a token

    def _mutate_permanent_move(self, e):
        p = self.players[e["active_idx"]]
        name, slot = e["permanent"]
        key = (name, slot)
        from_zone = _norm_zone(e.get("from_zone"))
        to_zone = _norm_zone(e.get("to_zone"))

        if to_zone == "battlefield":
            is_token = self._resolve_incoming_permanent(p, name, from_zone)
            entry = {
                "name": name, "tapped": bool(e.get("tapped")),
                "power": e.get("power"), "toughness": e.get("toughness"),
                # Printed stats at zone-entry, kept alongside the live
                # "power"/"toughness" (which stats_changed updates as counters/
                # Auras/pump come and go) so the viewer can highlight a
                # creature that's currently buffed/debuffed off its base.
                "base_power": e.get("power"), "base_toughness": e.get("toughness"),
                "is_token": is_token, "card_type": e.get("card_type"),
                "attacking": False, "blocking": None, "front_name": None,
            }
            p.battlefield[key] = entry
            return f"P{p.idx}: {name} enters the battlefield" + (" (token)" if is_token else "")

        old = p.battlefield.pop(key, None)
        if old is None:
            return f"P{p.idx}: {name} leaves the battlefield"
        if to_zone == "ceases_to_exist" or old.get("is_token"):
            return f"P{p.idx}: {name} ceases to exist"
        dst_pool = {"hand": p.hand, "graveyard": p.graveyard, "exile": p.exile}.get(to_zone)
        if dst_pool is not None:
            dst_pool.append({"name": old.get("front_name") or name})
        return f"P{p.idx}: {name} → {to_zone or '?'}"

    def _pop_named_source(self, p, name, from_zone):
        """Best-effort remove `name` from wherever the log says it came from,
        so a card's origin zone stops showing it as still present. Falls
        through silently if untracked (straight from the hidden library --
        nothing to remove). The candidate-pool search order for a
        from_zone=None cast (Flashback/Escape/Madness/Plot/Adventure/an
        eagerly-discarded alt cost/a copy -- the log doesn't distinguish
        these further) is what avoids the Lava Dart double-copy bug this
        class of lookup is prone to."""
        if from_zone == "hand":
            pop_by_name(p.hand, name)
        elif from_zone == "stack":
            self._pop_stack(name)
        elif from_zone in ("graveyard", "exile"):
            pop_by_name({"graveyard": p.graveyard, "exile": p.exile}[from_zone], name)
        elif from_zone is None:
            if pop_by_name(p.exile, name) is not None:
                return
            for i, c in enumerate(p.pending_resolution):
                if c["name"] == name:
                    p.pending_resolution.pop(i)
                    return
            pop_by_name(p.graveyard, name)

    def _mutate_card_move(self, e, name):
        from_zone = _norm_zone(e.get("from_zone"))
        to_zone = _norm_zone(e.get("to_zone"))

        if to_zone == "stack":
            p = self.players[e.get("controller", e["active_idx"])]
            self._pop_named_source(p, name, from_zone)
            self.stack.append({"name": name, "controller": p.idx})
            return

        if from_zone == "stack" and to_zone is None:
            # Resolution marker only, no destination yet: claimed by a
            # battlefield entry or a same-phase recast for the same name,
            # else flushed to graveyard at the next turn/phase boundary.
            p = self.players[e["active_idx"]]
            if self._pop_stack(name) is not None:
                p.pending_resolution.append({"name": name})
            return

        p = self.players[e["active_idx"]]
        self._pop_named_source(p, name, from_zone)
        dst_pool = {"hand": p.hand, "graveyard": p.graveyard, "exile": p.exile}.get(to_zone)
        if dst_pool is not None:
            dst_pool.append({"name": name})
        # library / library_bottom: not tracked as a distinct pool here.
        reason = e.get("reason")
        if reason == "draw":
            p.library_draws += 1
        elif reason in ("mulligan_take", "mulligan_bottom"):
            # A mulligan reject/bottom returns a drawn card to the library --
            # give the count back so a multi-mulligan pregame still nets to
            # the real kept-hand size, not an inflated one.
            p.library_draws = max(0, p.library_draws - 1)

    _HANDLERS = {
        "turn_start": _handle_turn_start,
        "phase_change": _handle_phase_change,
        "pass": _handle_pass,
        "zone_move": _handle_zone_move,
        "mana_tap": _handle_mana_tap,
        "mana_spend": _handle_mana_spend,
        "untap_step": _handle_untap_step,
        "mana_emptied": _handle_mana_emptied,
        "attack_declared": _handle_attack_declared,
        "block_assigned": _handle_block_assigned,
        "aura_attached": _handle_aura_attached,
        "aura_orphaned": _handle_aura_orphaned,
        "life_change": _handle_life_change,
        "state_based_death": _handle_state_based_death,
        "destroy": _handle_destroy,
        "countered": _handle_countered,
        "put_on_top": _handle_put_on_top,
        "mill": _handle_mill,
        "graveyard_exiled": _handle_graveyard_exiled,
        "graveyards_exiled": _handle_graveyards_exiled,
        "impulse_exile": _handle_impulse_exile,
        "tap": _handle_tap,
        "untap": _handle_untap,
        "tap_or_untap": _handle_tap_or_untap,
        "tuck": _handle_tuck,
        "transform": _handle_transform,
        "stats_changed": _handle_stats_changed,
        "reveal": _handle_reveal,
        "decision_weights": _handle_decision_weights,
    }


def list_games(doc):
    """Lightweight per-game index for the file-picker step: label + event
    count, no board-state reduction (cheap even for a multi-thousand-game
    round-robin --eval log).

    Prefers each game's own deck_a/deck_b (run_league.py's _write_event_log,
    2026-08 on) to label it directly as "A vs B" -- a round-robin log holds
    many different pairings in one file, so a single file-level label can't
    describe every game. A per-pairing occurrence counter disambiguates
    repeat games of the same pairing (a double round-robin plays each one
    twice) as "(game 1)"/"(game 2)" instead of two identical, unindexable
    labels. Falls back to the old file-level meta (matchup/deck_a, else a
    bare "game N") for logs written before per-game pairing existed.

    header/detail split the label into the two pieces the game-picker UI
    renders on separate lines (participants as the prominent line, game
    number + event count as the secondary one); label is header+detail
    joined back into one string, kept for callers that just want text."""
    meta = doc.get("meta") or {}
    matchup = meta.get("matchup")
    if meta.get("config_name"):
        base_label = meta["config_name"]
    elif matchup:
        base_label = " vs ".join(matchup)
    elif meta.get("deck_a"):
        base_label = f'{meta["deck_a"]} vs {meta.get("deck_b", "?")}'
    else:
        base_label = None

    games = []
    pairing_occurrence = {}
    for i, g in enumerate(doc.get("games") or []):
        idx = g.get("game_index", i)
        n = len(g.get("events") or [])
        deck_a, deck_b = g.get("deck_a"), g.get("deck_b")
        if deck_a and deck_b:
            pairing_occurrence[(deck_a, deck_b)] = pairing_occurrence.get((deck_a, deck_b), 0) + 1
            header = f'{deck_a} vs {deck_b}'
            detail = f'game {pairing_occurrence[(deck_a, deck_b)]}, {n} events'
        else:
            header = base_label or "game"
            detail = f'game {idx}, {n} events'
        games.append({
            "game_index": idx, "header": header, "detail": detail,
            "label": f'{header} ({detail})', "num_events": n,
        })
    return {"meta": meta, "games": games}


def reduce_game(doc, game_index):
    """Reduce one game's events into a list of board-state steps."""
    games = doc.get("games") or []
    game = next((g for g in games if g.get("game_index") == game_index), None)
    if game is None:
        raise ValueError(f"game index {game_index} not found in this log")
    events = game.get("events") or []
    if events and not any(e.get("kind") == "turn_start" for e in events):
        raise ValueError(
            "this log doesn't look like an event-stream game (no turn_start events found) -- "
            "only the current event-stream log format is supported"
        )
    return {"game_index": game_index, "steps": GameReducer(events).run()}
