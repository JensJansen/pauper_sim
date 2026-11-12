"""Layer 7 -- pointer alignment. A pointer action is "token i". It is only meaningful if token i's row describes the
object its identity handle really is, and the legal set is exactly what the engine says it is."""

from support import corpus as ec


def test_every_identity_handle_matches_its_own_row(frames, vocab):
    """name, zone and side of the row == name, zone and owner of the object the handle points at."""
    n = 0
    for k, f in enumerate(frames):
        for i, d in enumerate(f.descriptors):
            if d is None:
                continue
            kind, zone, name, owner = d
            row = f.rows[i]
            assert f.vocab_idx[i] == vocab.index(name), f"frame {k} token {i}: {name} vs vocab idx {f.vocab_idx[i]}"
            assert row[ec.COL[f"zone[{zone}]"]] == 1.0, f"frame {k} token {i}: handle is in {zone}, row says otherwise"
            assert owner is not None, f"frame {k} token {i}: handle {d} not found in any zone"
            assert row[ec.COL["side_mine"]] == float(owner == f.seat), f"frame {k} token {i}: side flag vs owner"
            n += 1
    assert n > 5000


def test_tokens_without_a_handle_are_only_exile_hand_or_revealed(frames):
    ok_zones = [ec.COL[f"zone[{z}]"] for z in ("exile", "hand", "revealed")]
    for k, f in enumerate(frames):
        for i, d in enumerate(f.descriptors):
            if d is None:
                assert any(f.rows[i, c] == 1.0 for c in ok_zones), (
                    f"frame {k} token {i}: battlefield/graveyard/stack token without a handle")


# pointer category -> (handle kind, zone, side relative to the deciding seat) a legal target must have
EXPECTED_TARGET = {
    "declare_attackers": ("permanent", "battlefield", "mine"),
    "declare_blockers": ("permanent", "battlefield", "mine"),
    "choose_opponent_permanent": ("permanent", "battlefield", "theirs"),
    "mana_subdecision": ("permanent", "battlefield", "mine"),
    "choose_graveyard_card": (None, None, None),     # either graveyard, or a revealed hand
    "choose_stack_target": ("stack_entry", "stack", None),
    "choose_any_target": ("permanent", "battlefield", None),
    "choose_permanent": ("permanent", "battlefield", "mine"),
    "assign_combat_damage": ("permanent", "battlefield", None),
}


def test_legal_pointer_targets_are_the_right_kind_of_object(frames):
    checked = 0
    for k, f in enumerate(frames):
        if f.pointer_kind is None:
            continue
        kind, zone, whose = EXPECTED_TARGET.get(f.pointer_kind, (None, None, None))
        for i in f.full_mask[f.n_fixed:].nonzero()[0]:
            d = f.descriptors[i]
            assert d is not None, f"frame {k} ({f.pointer_kind}): legal token {i} has no handle"
            if kind is not None:
                assert d[0] == kind and d[1] == zone, f"frame {k} ({f.pointer_kind}): legal target is {d[:2]}"
            if whose is not None:
                assert (d[3] == f.seat) == (whose == "mine"), f"frame {k} ({f.pointer_kind}): target on the wrong side"
            checked += 1
    assert checked > 500


def test_legal_pointer_count_equals_the_engines_option_count(frames):
    n = 0
    for k, f in enumerate(frames):
        if f.engine_option_count is None:
            continue
        got = int(f.full_mask[f.n_fixed:].sum())
        assert got == f.engine_option_count, (f"frame {k} ({f.pointer_kind}): {got} legal pointer tokens vs "
                                              f"{f.engine_option_count} engine options")
        n += 1
    assert n > 300


def test_forced_decisions_have_exactly_one_legal_action(frames):
    for k, f in enumerate(frames):
        if f.forced:
            assert int(f.full_mask.sum()) == 1, f"frame {k}"


def test_the_agents_own_token_list_and_identity_list_are_the_tokenizers(frames):
    """What the network is fed (dec.tokens) and what a pointer index is resolved against (dec.identities) must be the
    very same sequence the tokenizer produces: same order, same rows, same objects."""
    checked = [(k, f.agent_aligned) for k, f in enumerate(frames) if f.agent_aligned is not None]
    assert len(checked) > 500
    bad = [k for k, ok in checked if not ok]
    assert not bad, f"{len(bad)} decisions where the agent's tokens/identities are misaligned, e.g. frame {bad[0]}"
