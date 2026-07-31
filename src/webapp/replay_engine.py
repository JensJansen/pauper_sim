"""Fold an engine event-stream game log (GameState.log_event format, see
game/state.py) directly into a sequence of board-state snapshots for the
replay viewer -- no intermediate file format, no protobuf, the webapp reads
the raw JSON log itself.

Ports the event-kind interpretation rules already proven out in
sim_replay_converter/convert.py's EventStreamReplayBuilder (name-based zone
identity tracking via pop-by-name, DFC face reverts, aura-orphan handling,
countered-spell routing, same-phase-recast/pending-resolution handling,
mulligan netting -- each one fixing a real bug found against real logged
games, e.g. the Lava Dart flashback identity bug). Kept as separate code
rather than sharing a module with convert.py: that pipeline is tested
against the Cockatrice wire protocol and refactoring it wasn't in scope
here. One deliberate difference from it: the stack is tracked as ONE shared
ordered list (top = last), matching the real GameState.stack, rather than
convert.py's per-player split (a Cockatrice-protocol artifact, not a rules
one) -- true LIFO order across both players is exactly the kind of fidelity
win over Cockatrice this viewer exists for.

Event kinds with no board-visible effect (pass, priority_flip,
resolution_begin/complete, combat_damage/fight_damage -- superseded by
life_change, which fires for the same damage with the total already
computed, trigger_fired, pump/explore/animated -- the log entry doesn't
carry enough to render unambiguously, see convert.py's process_events
comment, undercity/initiative/goad/ward status, ...) still produce a step
(so the scrubber timeline never silently skips an event) but don't mutate
board state, matching convert.py's own documented scope.
"""
from collections import Counter


def pop_by_name(pool, name):
    """Pop and return the first {"name": ...} entry matching name, else None."""
    for i, c in enumerate(pool):
        if c["name"] == name:
            return pool.pop(i)
    return None


def _norm_zone(z):
    """exile_untracked / exile_mesmeric / etc. are all just "exile" to a viewer."""
    return "exile" if z and z.startswith("exile") else z


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

    def snapshot(self):
        battlefield = sorted(self.battlefield.items(), key=lambda kv: kv[0][1])
        return {
            "life": self.life,
            "mana_pool": dict(self.mana_pool),
            "hand": [c["name"] for c in self.hand],
            "graveyard": [c["name"] for c in self.graveyard],
            "exile": [c["name"] for c in self.exile],
            "battlefield": [
                {
                    "name": entry["name"], "slot": slot, "tapped": entry["tapped"],
                    "power": entry.get("power"), "toughness": entry.get("toughness"),
                    "is_token": entry.get("is_token", False), "card_type": entry.get("card_type"),
                    "attacking": entry.get("attacking", False), "blocking": entry.get("blocking"),
                }
                for (_, slot), entry in battlefield
            ],
        }


class GameReducer:
    def __init__(self, events):
        self.players = [_Player(0), _Player(1)]
        self.stack = []  # [{"name": str, "controller": int}], top = last
        self.steps = []

        first_turn = next((i for i, e in enumerate(events) if e["kind"] == "turn_start"), len(events))
        self._net_opening_hands(events[:first_turn], events[first_turn] if first_turn < len(events) else {})
        self.main_events = events[first_turn:]

    def other(self, p):
        return self.players[1 - p.idx]

    # ------------------------------------------------------------------
    def _net_opening_hands(self, pregame_events, first_event):
        """Every mulligan attempt is logged as a fresh full-hand draw, with the
        cards not kept logged as mulligan_take/mulligan_bottom put-backs. Net
        these into the single kept opening hand rather than replaying every
        round (an agent that mulligans to zero would otherwise dump ~100 churn
        steps at t=0) -- mirrors convert.py's EventStreamReplayBuilder.__init__.
        """
        opening = {0: Counter(), 1: Counter()}
        draws = {0: 0, 1: 0}
        for e in pregame_events:
            if e["kind"] != "zone_move":
                continue
            names = e.get("cards") or ([e["card"]] if e.get("card") else [])
            reason = e.get("reason")
            idx = e["active_idx"]
            if reason == "draw":
                opening[idx] += Counter(names)
                draws[idx] += 1
            elif reason in ("mulligan_take", "mulligan_bottom"):
                opening[idx] -= Counter(names)

        for p in self.players:
            p.hand = [{"name": n} for n in opening[p.idx].elements()]

        mulligans = {idx: max(0, draws[idx] - 1) for idx in (0, 1)}
        if any(mulligans.values()):
            desc = f"Opening hands drawn (P0 mulliganed {mulligans[0]}x, P1 mulliganed {mulligans[1]}x)"
        else:
            desc = "Opening hands drawn"
        self._emit(first_event, "opening_hands", desc)

    def _emit(self, e, kind, description):
        self.steps.append({
            "index": len(self.steps),
            "kind": kind,
            "turn": e.get("turn"),
            "phase": e.get("phase"),
            "active_player_idx": e.get("active_idx"),
            "turn_player_idx": e.get("turn_player_idx"),
            "description": description,
            "players": [p.snapshot() for p in self.players],
            "stack": [dict(c) for c in self.stack],
        })

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
            aura["enchanting"] = list(e["target"])
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
            p.battlefield[(to_card, slot)] = entry
        self._emit(e, "transform", f"{from_name} transforms into {to_card or '?'}")

    def _handle_reveal(self, e):
        p = self.players[e["active_idx"]]
        self._emit(e, "reveal", f"P{p.idx} reveals {e.get('card') or '?'} off the top of their library")

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
        (deck order isn't tracked, matching convert.py); only the graveyard
        case is a real move."""
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
        from, and report whether this is a token (no prior identity anywhere).
        Mirrors convert.py's _resolve_incoming_permanent."""
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
        nothing to remove). Mirrors convert.py's pop-by-name identity
        tracking, including the same candidate-pool search order for a
        from_zone=None cast (Flashback/Escape/Madness/Plot/Adventure/an
        eagerly-discarded alt cost/a copy -- the log doesn't distinguish
        these further) that fixed the real Lava Dart double-copy bug there."""
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

    _HANDLERS = {
        "turn_start": _handle_turn_start,
        "phase_change": _handle_phase_change,
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
        "reveal": _handle_reveal,
    }


def list_games(doc):
    """Lightweight per-game index for the file-picker step: label + event
    count, no board-state reduction (cheap even for a multi-thousand-game
    round-robin --eval log)."""
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
    for i, g in enumerate(doc.get("games") or []):
        idx = g.get("game_index", i)
        n = len(g.get("events") or [])
        label = f'{base_label or "game"} — game {idx} ({n} events)'
        games.append({"game_index": idx, "label": label, "num_events": n})
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
