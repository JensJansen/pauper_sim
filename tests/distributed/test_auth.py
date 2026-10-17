"""Self-check for distributed.auth's credential/token primitives."""
from distributed import auth


def test_verify_registration_credential_accepts_matching_value():
    cred = auth.generate_registration_credential()
    assert auth.verify_registration_credential(cred, cred) is True


def test_verify_registration_credential_rejects_wrong_value():
    cred = auth.generate_registration_credential()
    assert auth.verify_registration_credential("not-it", cred) is False


def test_verify_registration_credential_rejects_non_string_input():
    # a malformed/missing request body shouldn't crash the comparison
    assert auth.verify_registration_credential(None, "expected") is False
    assert auth.verify_registration_credential(123, "expected") is False


def test_mint_deck_token_is_unique_across_many_calls():
    tokens = {auth.mint_deck_token() for _ in range(200)}
    assert len(tokens) == 200
