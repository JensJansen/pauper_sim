"""Layer 2 -- hidden information and symmetry. The metamorphic checks run on the LIVE state while the corpus is built
(they have to mutate it and restore it), so here we demand they all held and that enough of them ran, plus a negative
control on the mirror comparison."""
import numpy as np
import pytest

from support import corpus as ec
from rl.model import features as F


@pytest.mark.parametrize("flag,what", [
    ("hidden_ok", "reordering the opponent's hand and both libraries changed the observation"),
    ("symmetry_ok", "the other seat's view of the shared tokens is not this view mirrored"),
    ("slot_ok", "relabelling permanent slot numbers changed the observation"),
])
def test_metamorphic_property_holds_on_every_checked_decision(frames, flag, what):
    checked = [(k, getattr(f, flag)) for k, f in enumerate(frames) if getattr(f, flag) is not None]
    assert len(checked) >= 500, f"only {len(checked)} decisions were checked"
    failed = [k for k, ok in checked if not ok]
    assert not failed, f"{what}: {len(failed)}/{len(checked)} decisions, e.g. frame {failed[0]}"


def test_mirror_transform_is_an_involution_and_swaps_the_right_columns():
    row = np.zeros(F.TOKEN_FEATURE_DIM, dtype=np.float32)
    row[ec.COL["side_mine"]] = 1.0
    row[ec.COL["targeted_by_mine"]] = 1.0
    flipped = ec._flip(row)
    assert flipped[ec.COL["side_mine"]] == 0.0
    assert flipped[ec.COL["targeted_by_theirs"]] == 1.0 and flipped[ec.COL["targeted_by_mine"]] == 0.0
    assert (ec._flip(flipped) == row).all()
