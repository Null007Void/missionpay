"""Shared test fixtures and helpers.

``accept_interpretation`` exists because the AI stage is now a persisted precondition
of review: a claim cannot be authorized until an ACCEPTED interpretation is recorded
for its exact evidence. Tests that drive the money path must therefore say what the
reviewer reviewed, instead of the old behaviour where an interpretation was assumed.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_credential_file(monkeypatch, tmp_path_factory):
    """No test may read the credential file saved on this machine.

    MissionPay reads ``%LOCALAPPDATA%/missionpay/credentials.env`` whenever hosted or
    sandbox mode is configured, even when the key is supplied in the environment. A suite
    that inherits that path reads a real secret, and its results change depending on who
    runs it: one hosted test only passed while the developer's file did not yet contain a
    key. Every test therefore starts with the path pointing at a file that does not exist.
    A test that wants a credential file sets this variable itself, which wins.
    """
    monkeypatch.setenv(
        "MISSIONPAY_CREDENTIALS_FILE",
        str(tmp_path_factory.getbasetemp() / "no-such-credentials.env"),
    )


def accept_interpretation(
    store,
    claim_id: str,
    revision: int,
    evidence_sha256: str,
    *,
    model_id: str = "fixture/tests-v1",
    output_sha256: str = "1" * 64,
    config_sha256: str = "0" * 64,
):
    """Record one synthetic ACCEPTED interpretation through the real store API."""
    return store.record_interpretation(
        claim_id=claim_id,
        revision=revision,
        evidence_sha256=evidence_sha256,
        ai_mode="fixture",
        model_id=model_id,
        config_sha256=config_sha256,
        status="ACCEPTED",
        output_sha256=output_sha256,
        facts=[
            {
                "name": "total",
                "value": "USD 78.99",
                "epistemic_status": "observed",
                "start": 0,
                "end": 8,
                "quote": "synthetic",
            }
        ],
        missing=("independent proof of attendance",),
    )


def authorize(store, reviewer_token, claim_id, *, revision=1, record=None):
    """Approve against the exact interpretation record a reviewer would have seen."""
    if record is None:
        raise AssertionError("tests must name the interpretation they approved")
    return store.authorize_claim(
        reviewer_token,
        claim_id,
        revision=revision,
        interpretation_id=record["interpretation_id"],
        interpretation_output_sha256=record["output_sha256"],
    )


def interpret_and_authorize(store, reviewer_token, claim_id, *, revision=1, evidence_sha256,
                            model_id="fixture/tests-v1"):
    record = accept_interpretation(store, claim_id, revision, evidence_sha256, model_id=model_id)
    return authorize(store, reviewer_token, claim_id, revision=revision, record=record), record


def approval_fields(page_html: str) -> dict:
    """The review identity a browser would submit with the approval form.

    Tests must use this rather than inventing values, so they exercise the same
    hidden fields a reviewer's stale page would carry.
    """
    import re

    fields = {}
    for name in ("interpretation_id", "interpretation_output_sha256"):
        match = re.search(rf'name="{name}"\s+value="([^"]+)"', page_html, re.S)
        if match:
            fields[name] = match.group(1)
    return fields
