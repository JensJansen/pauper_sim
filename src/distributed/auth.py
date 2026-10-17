"""Credential/token primitives for the hub. Two distinct secrets, not one:
a registration credential (shared, baked into every node image, good for
exactly one thing -- registering) and per-deck tokens (minted fresh by the
hub at registration time, scoped to a single deck's write access). See
DISTRIBUTED_TRAINING_PLAN.md's "Node lifecycle" for why they're split.

Stdlib-only (`secrets`), no repo imports -- this module has nothing to do
with training and should never need torch.
"""
import secrets


def generate_registration_credential():
    """Run once by the hub operator; the result goes into every node's
    provisioning template. Not stored by this module -- the caller decides
    where it lives (out of band, distributed to node images)."""
    return secrets.token_hex(32)


def verify_registration_credential(provided, expected):
    """Constant-time compare -- timing-attack resistance is free here, no
    reason to use a plain `==` instead."""
    if not isinstance(provided, str) or not isinstance(expected, str):
        return False
    return secrets.compare_digest(provided, expected)


def mint_deck_token():
    """A fresh, opaque token for one deck claim. Same primitive as
    generate_registration_credential (secrets.token_hex(32)) -- kept as a
    separate function since the two serve different purposes and callers
    should never conflate them."""
    return secrets.token_hex(32)


if __name__ == "__main__":
    # `python -m distributed.auth` -- prints one fresh registration
    # credential to stdout. The operator's only real interaction with this
    # module: generate once, put the value into every node's provisioning
    # template and into the hub's own credential file (see hub_server.py's
    # --credential-file). Deliberately no file-writing here -- where this
    # value lives (a gitignored node-local config, a provisioning secret
    # store) is an operational decision, not this module's to make.
    print(generate_registration_credential())
