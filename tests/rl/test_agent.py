"""Tests for rl.agent: SeatAgent's mulligan-vs-policy dispatch routing."""
import random
import types

import numpy as np
import pytest
import torch

import drl_env
import game
from game.state import CardInstance, GameState, PlayerState
from rl.action_bridge import build_fixed_action_table
from rl.agent import (
    DECK_SIZE_CAP, OPPONENT_HAND_SIZE_CAP, AlwaysKeep, SeatAgent, _Decision,
    _log_decision_weights, _resolve_pointer_identity, _scalar_features,
)
# HeuristicAgent moved to its own module 2026-08-17. Its tests stayed in this
# file rather than moving with it: they are built on _rally_ctx / _make_creature
# / TOKEN_DEFS, fixtures shared with the SeatAgent tests here, and duplicating
# those to achieve a tidier file split would be the more fragile arrangement.
from rl.heuristic_agent import (
    HeuristicAgent, _attack_is_worthwhile, _blocker_worth_assigning,
    _card_name_from_cast_action, _cmc,
)
from rl.arch import SetTransformer
from rl.deck import DeckNetwork
from rl.features import CardVocab
from rl.mulligan import MulliganNet


def _mulligan_decision_state():
    """A REAL state parked at the first mulligan decision (advance the
    coroutine one step) -- no guessing at GameState internals."""
    decklist = game.parse_decklist_file("../data/mono_red_madness.txt")
    vocab = CardVocab([decklist])
    state = game.turn.new_multiplayer_game_state([decklist, decklist], starting_player_idx=0, rng=random.Random(0))
    gen = game.turn.game_coroutine(state)
    next(gen)
    assert state.pending_resolution is not None and state.pending_resolution["kind"] == "mulligan_decision", (
        f"expected first decision to be a mulligan, got {state.pending_resolution}"
    )
    return decklist, vocab, state


@pytest.mark.slow
def test_always_keep_routes_mulligan_decision():
    # AlwaysKeep routes mulligan_decision to a keep executor, records nothing.
    _decklist, vocab, state = _mulligan_decision_state()
    seat = state.active_idx
    ak_agent = SeatAgent(main=None, mulligan=AlwaysKeep(), deck_ctx=(vocab, []))
    dr = ak_agent.decide(state, seat, horizon=120, device="cpu")
    assert callable(dr.executor) and dr.ppo_entry is None and dr.mull_entry is None and not dr.is_pass


@pytest.mark.slow
def test_always_keep_refuses_mulligan_bottom():
    # AlwaysKeep must refuse mulligan_bottom (unreachable under keep-always).
    raised = False
    try:
        AlwaysKeep().decide(types.SimpleNamespace(pending_resolution={"kind": "mulligan_bottom"}))
    except AssertionError as e:
        raised = "mulligan_bottom" in str(e)
    assert raised, "AlwaysKeep must assert on mulligan_bottom"


@pytest.mark.slow
def test_mulligan_net_branch_routes_and_records():
    # MulliganNet branch: routes to the net and records a 'decision' transition.
    _decklist, vocab, state = _mulligan_decision_state()
    seat = state.active_idx
    shared = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)
    mnet = MulliganNet(shared, hidden=16)
    net_agent = SeatAgent(main=None, mulligan=mnet, deck_ctx=(vocab, []))
    dr2 = net_agent.decide(state, seat, 120, "cpu")
    assert callable(dr2.executor) and dr2.ppo_entry is None
    assert dr2.mull_entry is not None and dr2.mull_entry[0] == "decision", (
        f"MulliganNet decision must record a 'decision' transition, got {dr2.mull_entry}"
    )


@pytest.mark.slow
def test_main_fixed_table_has_no_pregame_action():
    # Fixed-table invariant: the main net's fixed table has ZERO pregame actions,
    # so in a pregame state its legal mask is all-False and the SeatAgent MUST
    # intercept before ever reaching _seat_step. Guards against a future re-add
    # of the actions.
    decklist, _vocab, state = _mulligan_decision_state()
    ftable = build_fixed_action_table(decklist)
    assert not any(drl_env.legal_action_mask(state, ftable)), (
        "regression: the main net's fixed table has a legal action in a pregame state -- "
        "pregame actions must stay removed (the mulligan model owns them)"
    )


@pytest.mark.slow
def test_scalar_features_library_hand_size_and_stack_targets():
    # _scalar_features: library size (mine/opponent), opponent hand size, and
    # stack-targets-me/opponent (the "player" target kind -- the one case
    # with no token to carry a bit on; see rl.features._stack_target_map and
    # its own build_token_set self-check for the three OBJECT target kinds).
    sf_seat0 = PlayerState(on_the_play=True)
    sf_seat0.library = [game.CARD_DEFS["Mountain"]] * 40
    sf_seat0.hand = [game.CARD_DEFS["Lightning Bolt"]] * 3
    sf_seat1 = PlayerState(on_the_play=False)
    sf_seat1.library = [game.CARD_DEFS["Mountain"]] * 10
    sf_seat1.hand = [game.CARD_DEFS["Lightning Bolt"]] * 5
    sf_state = GameState(on_the_play=True, players=[sf_seat0, sf_seat1])
    # seat 1's Lava Dart targets seat 0 as a PLAYER (controller=1).
    sf_state.stack = [{"card_def": game.CARD_DEFS["Lava Dart"], "resolve": None, "controller": 1,
                        "targets": (("player", 0),)}]

    sf0 = _scalar_features(sf_state, 0, horizon=40)
    sf1 = _scalar_features(sf_state, 1, horizon=40)
    # Tail layout (see _scalar_features's own append order): [..., my_library,
    # opp_library, opp_hand, stack_targets_me, stack_targets_opponent,
    # on_the_play, opp_mulligans, opp_cleanup_discards]. The last three were
    # appended 2026-08-13, which shifted every index here by 3 -- these are
    # counted from the END on purpose, so a future append shifts them again.
    my_lib_i, opp_lib_i, opp_hand_i, targets_me_i, targets_opp_i = -8, -7, -6, -5, -4
    assert sf0[my_lib_i] == 40 / DECK_SIZE_CAP and sf0[opp_lib_i] == 10 / DECK_SIZE_CAP
    assert sf1[my_lib_i] == 10 / DECK_SIZE_CAP and sf1[opp_lib_i] == 40 / DECK_SIZE_CAP, (
        "library-size scalars must flip with perspective, not stay pinned to seat 0/1"
    )
    assert sf0[opp_hand_i] == 5 / OPPONENT_HAND_SIZE_CAP, "seat 0's opponent-hand-size view must be seat 1's hand"
    assert sf1[opp_hand_i] == 3 / OPPONENT_HAND_SIZE_CAP, "seat 1's opponent-hand-size view must be seat 0's hand"
    # Lava Dart (controller=1) targets seat 0 as a player -- seat 0 sees
    # "stack targets me", seat 1 sees "stack targets opponent", never both.
    assert sf0[targets_me_i] == 1.0 and sf0[targets_opp_i] == 0.0
    assert sf1[targets_me_i] == 0.0 and sf1[targets_opp_i] == 1.0


@pytest.mark.slow
def test_log_decision_weights_sorts_top_candidates_and_respects_event_log_gate():
    # Fabricated decision (no real forward pass needed): 3 fixed candidates,
    # probabilities out of sorted order on purpose -- topk must re-sort.
    fixed_table = [("Pass", None, None), ("Cast Lightning Bolt", None, None), ("Play Mountain", None, None)]
    dec = _Decision(tokens=None, scalar=None, full_mask=np.array([True, True, True]),
                     identities=[], fixed_table=fixed_table, n_fixed=3, n_legal=3, sole_action=None)
    dist = torch.distributions.Categorical(probs=torch.tensor([[0.1, 0.7, 0.2]]))
    value = torch.tensor([0.42])
    full_mask_t = torch.tensor([[True, True, True]])

    state = GameState(on_the_play=True, players=[PlayerState(on_the_play=True), PlayerState(on_the_play=False)],
                       event_log=[])
    _log_decision_weights(state, dec, dist, value, action_idx=1, full_mask_t=full_mask_t)
    assert len(state.event_log) == 1
    ev = state.event_log[0]
    assert ev["kind"] == "decision_weights" and ev["network"] == "main"
    assert ev["chosen_index"] == 1 and abs(ev["value_estimate"] - 0.42) < 1e-6
    probs = [c["probability"] for c in ev["candidates"]]
    assert probs == sorted(probs, reverse=True), "candidates must be highest-probability first"
    assert ev["candidates"][0]["fixed_label"] == "Cast Lightning Bolt"  # index 1, prob 0.7 -- the highest

    # gate: event_log=None (ordinary training collection) -> zero-cost no-op
    state2 = GameState(on_the_play=True, players=[PlayerState(on_the_play=True), PlayerState(on_the_play=False)],
                        event_log=None)
    _log_decision_weights(state2, dec, dist, value, action_idx=1, full_mask_t=full_mask_t)
    assert state2.event_log is None


@pytest.mark.slow
def test_resolve_pointer_identity_finds_controller_by_membership_search():
    # No shortcut like "check which list a pending references" works in
    # general (Rooftop Percher pools BOTH graveyards into one pending) -- the
    # resolver must find the real owner by membership, for either side.
    p0, p1 = PlayerState(on_the_play=True), PlayerState(on_the_play=False)
    state = GameState(on_the_play=True, players=[p0, p1])

    bear = game.Permanent(game.CARD_DEFS["Balustrade Spy"])
    bear.slot = 1
    p1.battlefield.append(bear)  # P1's own creature

    grave_card = CardInstance(game.CARD_DEFS["Lightning Bolt"])
    p0.graveyard.append(grave_card)  # P0's own graveyard

    stack_entry = {"card_def": game.CARD_DEFS["Counterspell"], "controller": 1}

    assert _resolve_pointer_identity(state, None) is None
    assert _resolve_pointer_identity(state, bear) == {"name": "Balustrade Spy", "slot": 1, "controller": 1}
    assert _resolve_pointer_identity(state, grave_card) == {"name": "Lightning Bolt", "slot": None, "controller": 0}
    assert _resolve_pointer_identity(state, stack_entry) == {"name": "Counterspell", "slot": None, "controller": 1}


@pytest.mark.slow
def test_seat_step_logs_decision_weights_end_to_end_in_a_real_game():
    """Integration check against a real short self-play game -- exercises the
    full top-K + real-Permanent/stack-entry identity resolution path, not
    just fabricated inputs. A real game has plenty of forced (sole-legal)
    decisions too; those must never appear as decision_weights events."""
    from rl.rewards import action_count_win_reward_200_floor02
    from rl.train import _constant_pairing, collect_rollout

    decklist = game.parse_decklist_file("../data/mono_red_madness.txt")
    token_defs = (game.BLOOD_TOKEN_CARD_DEF, game.ROBOT_TOKEN_CARD_DEF)  # mono_red_madness can mint both
    vocab = CardVocab([decklist], token_card_defs=token_defs)
    fixed_table = build_fixed_action_table(decklist, token_card_defs=token_defs)
    deck_ctx = (vocab, fixed_table)
    shared = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)
    net = DeckNetwork(shared, film_condition_dim=16, non_targeting_n_actions=len(fixed_table), trunk_hidden=(16, 16))
    agent = SeatAgent(net, AlwaysKeep(), deck_ctx)
    pairing = _constant_pairing([agent, agent], [decklist, decklist],
                                 [action_count_win_reward_200_floor02] * 2, ["m", "m"])

    game_logs = []
    collect_rollout(pairing, 1, horizon=20, rng=random.Random(0), device="cpu", game_logs=game_logs)
    events = game_logs[0]
    dweights = [e for e in events if e["kind"] == "decision_weights"]
    assert dweights, "expected at least one real (non-forced) decision to have logged weights"
    for ev in dweights:
        assert ev["network"] == "main"
        assert 1 <= len(ev["candidates"]) <= 5
        probs = [c["probability"] for c in ev["candidates"]]
        assert probs == sorted(probs, reverse=True)
        for c in ev["candidates"]:
            assert (c["fixed_label"] is None) != (c["pointer_identity"] is None), (
                "exactly one of fixed_label/pointer_identity per candidate"
            )

    # same game, no event_log -- ordinary training collection logs nothing
    collect_rollout(pairing, 1, horizon=20, rng=random.Random(0), device="cpu")  # must not crash without game_logs


def _mono_red_rally_deck_ctx():
    from rl.pool import TOKEN_DEFS  # mono_red_rally mints Treasure -- needs the full token/pseudo-card set
    decklist = game.parse_decklist_file("../data/mono_red_rally.txt")
    vocab = CardVocab([decklist], token_card_defs=TOKEN_DEFS)
    fixed_table = build_fixed_action_table(decklist, token_card_defs=TOKEN_DEFS)
    return decklist, (vocab, fixed_table)


@pytest.mark.slow
def test_heuristic_agent_plays_real_mono_red_rally_mirror_games_without_crashing():
    """The gauntlet's tier-1 opponent, scoped to mono_red_rally per the
    owner's own choice. Real mirror games at a real horizon (not the
    trimmed horizon=20 used elsewhere in this file) specifically to reach
    combat and casting decisions repeatedly across many games -- the
    HeuristicAgent's own scoring code (attack-safety, block-to-kill) is
    exercised for real here, not just imported successfully. No crash across
    5 real games is the bar; behavioral quality isn't asserted (the owner's
    own rules are approximate by design)."""
    from rl.rewards import action_count_win_reward_200_floor02
    from rl.train import _constant_pairing, collect_rollout

    decklist, deck_ctx = _mono_red_rally_deck_ctx()
    agent = HeuristicAgent(deck_ctx)
    pairing = _constant_pairing([agent, agent], [decklist, decklist],
                                 [action_count_win_reward_200_floor02] * 2, [None, None])

    game_logs = []
    _bufs, _mull, played = collect_rollout(pairing, 5, horizon=120, rng=random.Random(0), device="cpu",
                                           record=False, game_logs=game_logs)
    assert played == 5
    for one_game_events in game_logs:
        game_over = [e for e in one_game_events if e["kind"] == "game_over"]
        assert len(game_over) == 1
        assert game_over[0]["winner"] in (0, 1, None)
    kinds_seen = {e["kind"] for ev in game_logs for e in ev}
    assert "combat_damage" in kinds_seen or "attack_declared" in kinds_seen, (
        "an aggressive deck's own mirror games over 5 real playouts should reach combat at least once -- "
        "otherwise the attack-safety scoring path was never actually exercised by this test"
    )


@pytest.mark.slow
def test_heuristic_agent_plays_real_mono_red_rally_vs_trained_policy_without_crashing():
    """Same crash-safety bar as the mirror test, but against a real (tiny,
    untrained) SeatAgent -- exercises the two agent TYPES coexisting in one
    game (mixed dispatch through collect_rollout's own pairing contract),
    the actual gauntlet-mechanism use case (a trained policy vs. the
    heuristic reference), not just heuristic-vs-heuristic."""
    from rl.rewards import action_count_win_reward_200_floor02
    from rl.train import _constant_pairing, collect_rollout

    decklist, deck_ctx = _mono_red_rally_deck_ctx()
    vocab, fixed_table = deck_ctx
    shared = SetTransformer(vocab.size, d_model=16, n_heads=2, n_layers=1, dim_feedforward=32)
    net = DeckNetwork(shared, film_condition_dim=16, non_targeting_n_actions=len(fixed_table), trunk_hidden=(16, 16))
    trained_agent = SeatAgent(net, AlwaysKeep(), deck_ctx)
    heuristic_agent = HeuristicAgent(deck_ctx)
    pairing = _constant_pairing([trained_agent, heuristic_agent], [decklist, decklist],
                                 [action_count_win_reward_200_floor02] * 2, [None, None])

    game_logs = []
    _bufs, _mull, played = collect_rollout(pairing, 5, horizon=120, rng=random.Random(0), device="cpu",
                                           record=False, greedy=True, game_logs=game_logs)
    assert played == 5
    for one_game_events in game_logs:
        game_over = [e for e in one_game_events if e["kind"] == "game_over"]
        assert len(game_over) == 1


def test_cmc_and_cast_action_name_parsing():
    bolt = game.CARD_DEFS["Lightning Bolt"]
    assert _cmc(bolt) == 1
    forest = game.CARD_DEFS["Forest"]
    assert _cmc(forest) == 0  # lands have cast_cost=None
    assert _card_name_from_cast_action("Cast Lightning Bolt") == "Lightning Bolt"
    assert _card_name_from_cast_action("Cast Lightning Bolt (free)") == "Lightning Bolt"


@pytest.mark.slow
def test_attack_is_worthwhile_follows_the_owners_cheapest_threat_rule():
    """A direct, isolated check of the owner's own stated rule (not just
    "plays a full game without crashing"): a 3-mana attacker facing a 3-mana
    blocker that can kill it is fine to attack with; facing a 1-mana blocker
    that can also kill it is not."""
    decklist, deck_ctx = _mono_red_rally_deck_ctx()
    vocab = deck_ctx[0]
    state = game.turn.new_multiplayer_game_state([decklist, decklist], starting_player_idx=0, rng=random.Random(0))
    state.active_idx = 0

    def _make_creature(state, side, name, power, toughness, cmc, tapped=False):
        card_def = game.CardDef(name, game.CardType.CREATURE, {"generic": cmc}, "test_creature",
                                 power=power, toughness=toughness)
        p = game.Permanent(card_def, tapped=tapped)
        state.players[side].battlefield.append(p)
        return p

    my_attacker = _make_creature(state, 0, "My Attacker", power=3, toughness=3, cmc=3)

    # No blockers at all -- safe.
    assert _attack_is_worthwhile(state, my_attacker) is True

    # A 3-mana blocker that can kill it -- fair trade, still worthwhile.
    threat_same_cost = _make_creature(state, 1, "Fair Blocker", power=3, toughness=1, cmc=3)
    assert _attack_is_worthwhile(state, my_attacker) is True

    # A cheaper 1-mana blocker that can ALSO kill it -- not worthwhile anymore.
    state.players[1].battlefield.remove(threat_same_cost)
    _make_creature(state, 1, "Cheap Blocker", power=3, toughness=1, cmc=1)
    assert _attack_is_worthwhile(state, my_attacker) is False


@pytest.mark.slow
def test_blocker_worth_assigning_checks_power_vs_attacker_toughness():
    decklist, deck_ctx = _mono_red_rally_deck_ctx()
    state = game.turn.new_multiplayer_game_state([decklist, decklist], starting_player_idx=0, rng=random.Random(0))
    state.active_idx = 1  # blocking happens from the defender's flipped perspective

    def _make_creature(state, side, name, power, toughness):
        card_def = game.CardDef(name, game.CardType.CREATURE, {"generic": 1}, "test_creature",
                                 power=power, toughness=toughness)
        p = game.Permanent(card_def)
        state.players[side].battlefield.append(p)
        return p

    attacker = _make_creature(state, 0, "Attacker", power=2, toughness=2)
    state.players[0].attackers.append(attacker)

    weak_blocker = _make_creature(state, 1, "Weak Blocker", power=1, toughness=4)
    assert _blocker_worth_assigning(state, weak_blocker) is False, "1 power can't kill a 2-toughness attacker"

    strong_blocker = _make_creature(state, 1, "Strong Blocker", power=2, toughness=1)
    assert _blocker_worth_assigning(state, strong_blocker) is True, "2 power meets the 2-toughness attacker's kill threshold"


# --- three public scalars the policy could not see (2026-08-13) ---


@pytest.mark.slow
def test_scalar_features_length_matches_the_declared_dim():
    """The one invariant that silently corrupts everything if broken: the
    vector _scalar_features builds must be exactly SCALAR_FEATURE_DIM long, or
    the trunk's first Linear is fed misaligned values with no error."""
    from rl.deck import SCALAR_FEATURE_DIM
    decklist = game.parse_decklist_file("../data/mono_red_madness.txt")
    state = game.new_multiplayer_game_state([decklist, decklist], 0, random.Random(0))
    for seat in (0, 1):
        vec = _scalar_features(state, seat, horizon=20)
        assert len(vec) == SCALAR_FEATURE_DIM, f"seat {seat}: {len(vec)} != {SCALAR_FEATURE_DIM}"
        assert all(0.0 <= v <= 1.0 for v in vec), "every scalar is cap-then-normalized into [0, 1]"


@pytest.mark.slow
def test_on_the_play_is_seat_specific_and_actually_reaches_the_policy():
    """It was a plain oversight: MulliganNet has always received on_the_play
    while the main policy never did. Exactly one seat is on the play."""
    from rl.deck import SCALAR_FEATURE_DIM
    decklist = game.parse_decklist_file("../data/mono_red_madness.txt")
    state = game.new_multiplayer_game_state([decklist, decklist], 0, random.Random(0))
    idx = SCALAR_FEATURE_DIM - 3  # on_the_play, then opp mulligans, then opp cleanup discards
    seat0 = _scalar_features(state, 0, horizon=20)[idx]
    seat1 = _scalar_features(state, 1, horizon=20)[idx]
    assert {seat0, seat1} == {0.0, 1.0}, f"exactly one seat is on the play, got {seat0}/{seat1}"
    assert seat0 == 1.0, "starting_player_idx=0 means seat 0 is on the play"


@pytest.mark.slow
def test_opponent_mulligans_and_cleanup_discards_are_read_from_the_OTHER_seat():
    """Both are public information in real Magic (you watch an opponent
    mulligan, and cleanup discards happen in the open), which is what makes
    them compliant observations rather than hidden-information leaks. They must
    read the opponent's counters, not the reader's own."""
    from rl.deck import SCALAR_FEATURE_DIM
    decklist = game.parse_decklist_file("../data/mono_red_madness.txt")
    state = game.new_multiplayer_game_state([decklist, decklist], 0, random.Random(0))
    state.players[1].mulligans_taken = 3
    state.players[1].cleanup_discard_turns = 5
    mull_idx, discard_idx = SCALAR_FEATURE_DIM - 2, SCALAR_FEATURE_DIM - 1

    seat0 = _scalar_features(state, 0, horizon=20)  # opponent is player 1
    assert seat0[mull_idx] == pytest.approx(3 / 7)
    assert seat0[discard_idx] == pytest.approx(5 / 7)

    seat1 = _scalar_features(state, 1, horizon=20)  # opponent is player 0, untouched
    assert seat1[mull_idx] == 0.0 and seat1[discard_idx] == 0.0
