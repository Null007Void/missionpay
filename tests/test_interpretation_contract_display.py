"""What the page may say about a record it did not judge under today's rule.

Review finding at web.py:599 — the citation note was printed under *every* interpretation,
including records accepted before the uniqueness contract existed. That claims a check which
never ran on that evidence. A record now carries the contract it was judged under; a record
without one is described as historical, and the current-rule sentence is withheld.
"""

from __future__ import annotations

import json
import re
import sqlite3

import pytest
from fastapi.testclient import TestClient

from missionpay import web
from missionpay.ai import PROMPT_VERSION
from missionpay.domain import EvidenceIdentity
from missionpay.store import MissionPayStore
from tests.test_hosted_corrections import _runtime, _seed_claim

#: A quote that appears three times in the synthetic $5 receipt, so under the current
#: contract it could never be accepted. Perfect for a record that predates the rule.
REPEATED_QUOTE = "5.00"


def _fact(quote=REPEATED_QUOTE, name="total_paid", start=0, end=4):
    return {
        "name": name,
        "value": "USD 5.00",
        "epistemic_status": "observed",
        "start": start,
        "end": end,
        "quote": quote,
        "offsets_derived": False,
    }


def _record(store, claim_id, evidence_sha256, *, contract_version, quote=REPEATED_QUOTE):
    return store.record_interpretation(
        claim_id=claim_id,
        revision=1,
        evidence_sha256=evidence_sha256,
        ai_mode="fixture",
        model_id="fixture/display-v1",
        config_sha256="0" * 64,
        status="ACCEPTED",
        output_sha256="1" * 64,
        facts=[_fact(quote=quote)],
        missing=("independent proof of attendance",),
        contract_version=contract_version,
    )


def _page(store, claim_id, tmp_path) -> str:
    client = TestClient(web.create_app(store, config=_runtime(tmp_path)))
    login_csrf = re.search(r'name="csrf_token" value="([^"]+)"', client.get("/login").text).group(1)
    assert (
        client.post(
            "/login",
            data={
                "email": "volunteer@example.test",
                "password": "volunteer-passphrase",
                "csrf_token": login_csrf,
            },
            follow_redirects=False,
        ).status_code
        == 303
    )
    response = client.get(f"/claims/{claim_id}")
    assert response.status_code == 200
    return response.text


def test_a_historical_record_is_not_described_as_checked_under_the_current_rule(tmp_path):
    store, claim_id, digest = _seed_claim(tmp_path)
    record = _record(store, claim_id, digest, contract_version=None)
    assert record["contract_version"] is None

    page = _page(store, claim_id, tmp_path)

    # The claim the reviewer objected to must be absent for a record that never earned it.
    assert "appears exactly once" not in page
    # And the replacement must not invent what the old contract required: the record does not
    # know, so the page says the contract is unrecorded and the question is unverified.
    assert "the contract it was judged under is unrecorded" in page
    assert "whether each quote would also satisfy the current uniqueness rule is unverified" in page
    assert "did not require" not in page
    assert "which did not require a quote" not in page


def test_a_current_record_states_the_current_rule(tmp_path):
    store, claim_id, digest = _seed_claim(tmp_path)
    _record(store, claim_id, digest, contract_version=PROMPT_VERSION, quote="CANYON SUPPLY COOPERATIVE")

    page = _page(store, claim_id, tmp_path)

    assert "appears exactly once" in page
    assert "whether each quote would also satisfy the current uniqueness rule is unverified" not in page
    # And the semantic limit is still stated to the reviewer.
    assert "does not show that the quote supports the fact" in page


def test_a_record_from_an_older_named_contract_names_it_without_judging_it(tmp_path):
    """A v3 record is knowable, but what v3 required of a duplicate quote is still not ours to assert."""
    store, claim_id, digest = _seed_claim(tmp_path)
    _record(store, claim_id, digest, contract_version="missionpay-receipt-v3")

    page = _page(store, claim_id, tmp_path)

    assert "recorded under contract missionpay-receipt-v3" in page
    assert "not the contract in force today" in page
    assert "whether each quote would also satisfy the current uniqueness rule is unverified" in page
    # v3 did not enforce uniqueness, but the page must not narrate that as a property it knows.
    assert "did not require" not in page
    assert "appears exactly once" not in page


def test_the_historical_record_is_still_shown_as_evidence(tmp_path):
    """Honest labelling is not deletion: the facts a reviewer relied on stay visible."""
    store, claim_id, digest = _seed_claim(tmp_path)
    _record(store, claim_id, digest, contract_version=None)

    page = _page(store, claim_id, tmp_path)

    assert REPEATED_QUOTE in page, "the quoted text a previous reviewer saw must still render"
    assert "Total Paid" in page
    assert "USD 5.00" in page


def test_an_older_database_gains_a_nullable_contract_column(tmp_path):
    """Existing rows must keep NULL — that NULL is the signal the page reads."""
    store, claim_id, digest = _seed_claim(tmp_path)
    _record(store, claim_id, digest, contract_version=None)

    connection = sqlite3.connect(store.path)
    try:
        connection.execute("ALTER TABLE interpretations DROP COLUMN contract_version")
        connection.commit()
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(interpretations)").fetchall()
        }
        assert "contract_version" not in columns
    finally:
        connection.close()

    MissionPayStore(store.path).initialize()
    connection = sqlite3.connect(store.path)
    try:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(interpretations)").fetchall()
        }
        assert "contract_version" in columns
        values = [row[0] for row in connection.execute("SELECT contract_version FROM interpretations")]
    finally:
        connection.close()
    assert values == [None], "a pre-existing record must not be retro-labelled"


def test_a_stored_contract_version_survives_a_reload(tmp_path):
    store, claim_id, digest = _seed_claim(tmp_path)
    _record(store, claim_id, digest, contract_version=PROMPT_VERSION)
    reopened = MissionPayStore(store.path)
    reopened.initialize()
    history = reopened.interpretation_history(claim_id, 1)
    assert history[-1]["contract_version"] == PROMPT_VERSION
