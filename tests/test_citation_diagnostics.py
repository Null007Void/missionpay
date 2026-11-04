"""Citation refusal diagnostics: enough to diagnose, nothing to leak.

MP-AI-001 ended as `CITATION_MISMATCH` with the answer discarded, so nobody could say
which fact failed or how. These tests fix the shape of what is kept: a fact name, a
failure category, and bounded offsets and lengths. Never the quote, never the receipt,
never the raw response, never the key.

They also fail against the reviewed code, where every citation failure raised the same
bare code and carried no structure at all.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from missionpay.ai import (
    build_catalog,
    CITATION_FAILURES,
    FACT_NAMES,
    CitationRefusal,
    ModelInterpretationError,
    interpret_receipt_text,
)
from missionpay.domain import EvidenceIdentity
from missionpay.hosted_qualify import qualify
from missionpay.catalog import Catalog, Excerpt, text_digest
from missionpay.store import MissionPayStore, clean_diagnostic
from tests.catalogue_support import id_for
from tests.test_hosted_corrections import (
    EXPECTED,
    ScriptedTransport,
    _runtime,
    _seed_claim,
    answer,
    model,
)

#: A receipt that contains a value nobody should ever find in a diagnostic.
SECRET = "SYNTHETIC-CARD-4111-1111-1111"
TEXT = (
    "CANYON SUPPLY COOPERATIVE\n"
    "TOTAL PAID USD 5.00\n"
    f"reference {SECRET}\n"
)
EVIDENCE = EvidenceIdentity("a" * 64, "text/plain", "synthetic.txt")
TOTAL_START = TEXT.index("USD 5.00")
TOTAL_END = TOTAL_START + len("USD 5.00")


class _Scripted:
    model_id = "test-fixture/v1"

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete_json(self, system: str, user: str) -> str:  # noqa: ARG002
        return json.dumps(self.payload)


def _payload(*, start=None, end=None, quote=None, name="total_paid", excerpt_id=None):
    """A fact in the current schema.

    Callers still like to say "cite this span" or "cite this quote", because that is how the
    failure they are describing was phrased. This translates: an exact span that matches a
    catalogue entry yields its number, and anything else yields an id that cannot exist, which
    is the modern equivalent of "this does not identify a place in the receipt".
    """
    if excerpt_id is None:
        excerpt_id = 9999
        if isinstance(start, int) and isinstance(end, int) and quote is not None:
            for entry in build_catalog(TEXT).excerpts:
                if entry.start == start and entry.end == end and entry.text == quote:
                    excerpt_id = entry.excerpt_id
                    break
    return {
        "facts": [
            {
                "name": name,
                "value": "USD 5.00",
                "epistemic_status": "observed",
                "excerpt_id": excerpt_id,
            }
        ],
        "missing": ["independent proof of attendance"],
    }


def _tampered(text: str, excerpt: str) -> Catalog:
    at = text.index(excerpt)
    return Catalog(
        digest=text_digest(text),
        excerpts=(Excerpt(excerpt_id=1, start=at, end=at + len(excerpt), text=excerpt),),
    )


def _refusal(**kwargs) -> CitationRefusal:
    catalog = kwargs.pop("catalog", None)
    with pytest.raises(CitationRefusal) as excinfo:
        interpret_receipt_text(
            evidence=EVIDENCE, text=TEXT, model=_Scripted(_payload(**kwargs)), catalog=catalog
        )
    return excinfo.value


# -- the refusal categories -------------------------------------------------


def test_a_valid_citation_is_accepted_and_resolves_to_its_line():
    """Equivalent of "supplied offsets are honoured": MissionPay resolves, exactly."""
    line = f"TOTAL PAID USD 5.00"
    result = interpret_receipt_text(
        evidence=EVIDENCE, text=TEXT, model=_Scripted(_payload(excerpt_id=id_for(TEXT, line)))
    )
    fact = result.facts[0]
    assert fact.citation.quote == line
    assert TEXT[fact.citation.start : fact.citation.end] == line
    assert fact.citation.excerpt_id == id_for(TEXT, line)
    assert fact.offsets_derived is False


def test_an_absent_excerpt_is_reported_without_any_text():
    """A fact citing an identifier the catalogue never offered."""
    exc = _refusal(excerpt_id=9999)
    assert exc.code == "CITATION_MISMATCH"
    assert exc.diagnostic["category"] == "EXCERPT_UNKNOWN"
    assert exc.diagnostic["fact_name"] == "total_paid"
    assert exc.diagnostic["excerpt_id"] == 9999
    assert exc.diagnostic["evidence_length"] == len(TEXT)
    assert "USD 5.00" not in json.dumps(exc.diagnostic)
    assert SECRET not in json.dumps(exc.diagnostic)


def test_an_ambiguous_excerpt_reports_how_many_places_it_matched():
    """The ambiguity count still exists; it is now asserted at the layer that can produce it."""
    exc = _refusal(excerpt_id=1, catalog=_tampered(TEXT, "S"))
    assert exc.diagnostic["category"] == "QUOTE_AMBIGUOUS"
    assert exc.diagnostic["occurrences"] >= 2


def test_a_receipt_with_nothing_citable_refuses_every_id():
    """Equivalent of the empty-quote category: nothing can identify a place."""
    text = "5.00\n5.00\n"
    with pytest.raises(CitationRefusal) as excinfo:
        interpret_receipt_text(
            evidence=EVIDENCE, text=text, model=_Scripted(_payload(excerpt_id=1))
        )
    assert excinfo.value.diagnostic["category"] == "EXCERPT_UNKNOWN"


def test_a_non_integer_identifier_is_refused_without_echoing_it():
    exc = _refusal(excerpt_id="1")
    assert exc.diagnostic["category"] == "EXCERPT_UNKNOWN"
    assert exc.diagnostic.get("excerpt_id") is None
    assert "1" not in json.dumps(exc.diagnostic)


def test_every_category_is_in_the_declared_set():
    for kwargs in (
        dict(start=0, end=8, quote="nowhere to be found"),
        dict(start=0, end=1, quote="S"),
        dict(start=0, end=8, quote=""),
        dict(start=0, end=8, quote=None),
    ):
        exc = _refusal(**kwargs)
        assert exc.diagnostic["category"] in CITATION_FAILURES


# -- the diagnostic cannot leak ---------------------------------------------


def test_no_diagnostic_ever_contains_text_from_the_receipt_or_the_answer():
    # An absent quote, so the refusal path is the one being inspected.
    exc = _refusal(start=0, end=8, quote="CANYON SX")
    rendered = json.dumps(exc.diagnostic)
    for forbidden in (
        SECRET,
        "4111",
        "CANYON",
        "USD 5.00",
        "TOTAL PAID",
        "reference",
        "\n",
    ):
        assert forbidden not in rendered, forbidden


def test_the_refusal_is_still_a_plain_interpretation_error_with_the_same_code():
    # TEXT[0:2] is "CA", so quoting "CY" at that span is a genuine mismatch.
    exc = _refusal(start=0, end=2, quote="CY")
    assert isinstance(exc, ModelInterpretationError)
    assert str(exc) == "CITATION_MISMATCH"
    assert exc.code == "CITATION_MISMATCH"


def test_an_unknown_fact_name_is_not_echoed_verbatim():
    refusal = CitationRefusal(
        fact_name="something-the-model-invented", category="QUOTE_NOT_FOUND"
    )
    assert refusal.diagnostic["fact_name"] == "unrecognised"


def test_an_unknown_category_is_a_programming_error_not_a_substitute():
    with pytest.raises(ValueError):
        CitationRefusal(fact_name="total_paid", category="MADE_UP")
    with pytest.raises(ValueError):
        CitationRefusal(fact_name="total_paid", category="quote_mismatch")


def test_the_store_whitelist_drops_anything_that_is_not_a_name_or_a_number():
    hostile = {
        "code": "CITATION_MISMATCH",
        "fact_name": "total_paid",
        "category": "QUOTE_NOT_FOUND",
        "start": 35,
        "end": 43,
        "quote": f"USD 5.00 {SECRET}",
        "raw_response": "{...}",
        "evidence_text": TEXT,
        "api_key": "sk-should-never-be-stored",
        "quote_length": 8,
    }
    cleaned = clean_diagnostic(hostile)
    assert cleaned == {
        "code": "CITATION_MISMATCH",
        "fact_name": "total_paid",
        "category": "QUOTE_NOT_FOUND",
        "start": 35,
        "end": 43,
        "quote_length": 8,
    }
    rendered = json.dumps(cleaned)
    assert SECRET not in rendered and "sk-" not in rendered


def test_the_whitelist_rejects_text_where_a_name_belongs():
    assert clean_diagnostic({"fact_name": "USD 5.00 " + SECRET}) is None
    assert clean_diagnostic({"fact_name": "x" * 200}) is None
    assert clean_diagnostic({"start": True}) is None
    assert clean_diagnostic({"start": 10**12}) is None
    assert clean_diagnostic(None) is None
    assert clean_diagnostic({}) is None


@pytest.mark.parametrize(
    "field,value",
    [
        # Identifier-shaped but not declared: the regex version accepted all of these.
        ("code", "SyntheticSecret"),
        ("code", "CITATION_MISMATCH\n"),
        ("code", "citation_mismatch"),
        ("category", "SyntheticSecret"),
        ("category", "quote_mismatch"),
        ("category", "QUOTE_NOT_FOUND\n"),
        ("category", "SPAN_NOT_INTEGER_"),
        ("fact_name", "SyntheticSecret"),
        ("fact_name", "sk_live_abc123"),
        ("fact_name", "TOTAL_PAID"),
        ("fact_name", "total_paid "),
        ("fact_name", "CANYON_SUPPLY_COOPERATIVE"),
    ],
)
def test_only_declared_values_are_accepted(field, value):
    """Exact membership, not `^[A-Za-z_]+$`: a pattern accepts a secret as a name."""
    baseline = {
        "code": "CITATION_MISMATCH",
        "fact_name": "total_paid",
        "category": "QUOTE_NOT_FOUND",
        "start": 291,
        "end": 299,
    }
    assert clean_diagnostic(dict(baseline)) is not None
    hostile = dict(baseline)
    hostile[field] = value
    assert clean_diagnostic(hostile) is None, (field, value)


def test_a_diagnostic_without_its_declared_identifiers_is_dropped_whole():
    assert clean_diagnostic({"category": "QUOTE_NOT_FOUND", "fact_name": "total_paid"}) is None
    assert clean_diagnostic({"code": "CITATION_MISMATCH", "fact_name": "total_paid"}) is None
    assert clean_diagnostic({"code": "CITATION_MISMATCH", "category": "QUOTE_NOT_FOUND"}) is None
    # "unrecognised" is the one extra fact name that is legitimate.
    assert clean_diagnostic(
        {
            "code": "CITATION_MISMATCH",
            "category": "QUOTE_NOT_FOUND",
            "fact_name": "unrecognised",
        }
    ) == {
        "code": "CITATION_MISMATCH",
        "category": "QUOTE_NOT_FOUND",
        "fact_name": "unrecognised",
    }


# -- the journal keeps it ----------------------------------------------------


def test_a_citation_refusal_is_recorded_with_its_diagnostic_and_no_text(tmp_path):
    store, claim_id, _digest = _seed_claim(tmp_path)
    wrong = {
        "facts": [
            {
                "name": "total_paid",
                "value": "USD 5.00",
                "epistemic_status": "observed",
                "excerpt_id": 424242,
            }
        ],
        "missing": ["independent proof of attendance"],
    }
    outcome = qualify(
        store,
        _runtime(tmp_path),
        authorization_id="diag-1",
        claim_id=claim_id,
        expectation=EXPECTED,
        model=model(ScriptedTransport(answer(json.dumps(wrong)))),
    )
    assert outcome["status"] == "FAILED"
    assert outcome["code"] == "CITATION_MISMATCH"
    diagnostic = outcome["diagnostic"]
    assert diagnostic["category"] in CITATION_FAILURES
    assert diagnostic["fact_name"] == "total_paid"
    assert isinstance(diagnostic.get("start"), int) or diagnostic.get("start") is None
    assert diagnostic.get("excerpt_id") == 424242
    assert diagnostic["evidence_length"] == len(
        (Path(__file__).parent / "fixtures" / "hosted" / "live-receipt.txt").read_text(
            encoding="utf-8"
        )
    )

    row = store.hosted_attempt("diag-1")
    assert row["state"] == "FAILED"
    assert row["diagnostic"] == diagnostic
    assert row["usage"] is not None, "a refused answer still cost tokens"

    stored = json.dumps(row, sort_keys=True)
    receipt = (
        Path(__file__).parent / "fixtures" / "hosted" / "live-receipt.txt"
    ).read_text(encoding="utf-8")
    for forbidden in ("CANYON SUPPLY COOPERATIVE", "Safety whistle", "4412"):
        assert forbidden not in json.dumps(row["diagnostic"]), forbidden
    assert receipt[:40] not in stored


def test_an_older_journal_gains_the_diagnostic_column(tmp_path):
    store, _claim_id, _digest = _seed_claim(tmp_path)
    connection = sqlite3.connect(store.path)
    try:
        connection.execute("ALTER TABLE hosted_attempts DROP COLUMN diagnostic_json")
        connection.commit()
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(hosted_attempts)").fetchall()
        }
        assert "diagnostic_json" not in columns
    finally:
        connection.close()

    MissionPayStore(store.path).initialize()
    connection = sqlite3.connect(store.path)
    try:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(hosted_attempts)").fetchall()
        }
    finally:
        connection.close()
    assert "diagnostic_json" in columns


def test_the_audit_trail_records_the_refusal_shape(tmp_path):
    store, claim_id, _digest = _seed_claim(tmp_path)
    store.record_citation_refusal(
        claim_id,
        1,
        {
            "code": "CITATION_MISMATCH",
            "fact_name": "total_paid",
            "category": "QUOTE_NOT_FOUND",
            "start": 291,
            "end": 299,
            "quote": "CANYON SUPPLY COOPERATIVE",
        },
        model_id="openai:gpt-4.1-mini",
    )
    events = [
        event
        for event in store.audit_trail(claim_id)
        if event["event_type"] == "INTERPRETATION_CITATION_REFUSAL"
    ]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["diagnostic"]["start"] == 291
    assert "quote" not in payload["diagnostic"]
    assert "CANYON" not in json.dumps(payload)


# -- the operator path MP-AI-001 took ---------------------------------------


def test_the_cli_prints_the_diagnostic_and_writes_no_text(tmp_path, monkeypatch, capsys):
    """The command that spent the credit must show where it failed, and nothing else."""
    import argparse

    from missionpay import cli

    store, claim_id, _digest = _seed_claim(tmp_path)
    config = _runtime(tmp_path)
    wrong = {
        "facts": [
            {
                "name": "total_paid",
                "value": "USD 5.00",
                "epistemic_status": "observed",
                "excerpt_id": 424242,  # a number the catalogue never offered
            }
        ],
        "missing": ["independent proof of attendance"],
    }
    transport = ScriptedTransport(answer(json.dumps(wrong)))
    monkeypatch.setattr(cli, "_config", lambda: config)
    monkeypatch.setattr("missionpay.hosted_qualify.build_model", lambda _config: model(transport))

    out_json = tmp_path / "outcome.json"
    args = argparse.Namespace(
        dry_run=False,
        confirm=True,
        authorization="cli-diag",
        claim=claim_id,
        revision=1,
        expect=str(Path(__file__).parent / "fixtures" / "hosted"),
        json=str(out_json),
    )
    exit_code = cli._hosted_qualify(args)
    assert exit_code == 1, "no verdict"

    captured = capsys.readouterr()
    assert "diagnostic {" in captured.out
    assert "CITATION_MISMATCH" in captured.out
    written = out_json.read_text(encoding="utf-8")
    for forbidden in ("CANYON SUPPLY COOPERATIVE", "Safety whistle", "4412", "river cleanup"):
        assert forbidden not in captured.out, forbidden
        assert forbidden not in written, forbidden
    # The diagnostic is present in both places and identical.
    payload = json.loads(written)
    assert payload["diagnostic"]["fact_name"] == "total_paid"
    assert payload["diagnostic"]["category"] == "EXCERPT_UNKNOWN"
    assert payload["diagnostic"]["excerpt_id"] == 424242


# -- sanitised once, identical on every surface ----------------------------


def _hostile_exception():
    """An exception carrying text in every field a careless caller might set.

    This is the review's case: the diagnostic attribute is not private, so anything that
    raised could have put a quote, a nested payload or a secret in it. Qualify must
    sanitise before the object reaches *any* surface.
    """
    from missionpay.hosted import HostedModelError

    exc = HostedModelError("CITATION_MISMATCH")
    exc.diagnostic = {
        "code": "CITATION_MISMATCH",
        "fact_name": "total_paid",
        "category": "QUOTE_NOT_FOUND",
        "start": 291,
        "end": 299,
        "quote": "USD 5.00 CANYON SUPPLY COOPERATIVE",
        "nested": {"raw_response": "{\"facts\": []}", "api_key": "sk-not-real"},
        "evidence_text": "TOTAL PAID USD 5.00",
        "note": "SYNTHETIC-4111-1111-1111",
        "list": ["USD 5.00", SECRET],
        "quote_length": "8",
        "bad_code": "SyntheticSecret",
    }
    return exc


def test_one_sanitised_object_reaches_the_journal_the_outcome_and_the_cli(
    tmp_path, monkeypatch, capsys
):
    import argparse

    from missionpay import cli

    store, claim_id, _digest = _seed_claim(tmp_path)
    config = _runtime(tmp_path)
    transport = ScriptedTransport(_hostile_exception())
    monkeypatch.setattr(cli, "_config", lambda: config)
    monkeypatch.setattr("missionpay.hosted_qualify.build_model", lambda _config: model(transport))

    out_json = tmp_path / "outcome.json"
    args = argparse.Namespace(
        dry_run=False,
        confirm=True,
        authorization="sanitise-once",
        claim=claim_id,
        revision=1,
        expect=str(Path(__file__).parent / "fixtures" / "hosted"),
        json=str(out_json),
    )
    exit_code = cli._hosted_qualify(args)
    assert exit_code == 1

    captured = capsys.readouterr()
    written = json.loads(out_json.read_text(encoding="utf-8"))
    row = store.hosted_attempt("sanitise-once")

    expected = {
        "code": "CITATION_MISMATCH",
        "fact_name": "total_paid",
        "category": "QUOTE_NOT_FOUND",
        "start": 291,
        "end": 299,
    }
    assert row["diagnostic"] == expected, row["diagnostic"]
    assert written["diagnostic"] == expected
    assert json.loads(captured.out.split("diagnostic ", 1)[1].splitlines()[0]) == expected

    for surface in (json.dumps(row), json.dumps(written), captured.out):
        for forbidden in (
            SECRET,
            "4111",
            "CANYON",
            "USD 5.00",
            "TOTAL PAID",
            "sk-not-real",
            "raw_response",
            "SyntheticSecret",
            "evidence_text",
        ):
            assert forbidden not in surface, forbidden


def test_the_audit_event_carries_the_same_sanitised_object(tmp_path, monkeypatch):
    """The UI path sanitises at its own boundary and records that one object."""
    from fastapi.testclient import TestClient

    from missionpay import web

    store, claim_id, _digest = _seed_claim(tmp_path)
    config = _runtime(tmp_path)
    monkeypatch.setattr(
        web,
        "_interpret",
        lambda *_args: (
            None,
            "CITATION_MISMATCH",
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            clean_diagnostic(_hostile_exception().diagnostic),
        ),
    )
    app = web.create_app(store, config=config)
    client = TestClient(app)
    import re as _re

    login_csrf = _re.search(
        r'name="csrf_token" value="([^"]+)"', client.get("/login").text
    ).group(1)
    login = client.post(
        "/login",
        data={
            "email": "volunteer@example.test",
            "password": "volunteer-passphrase",
            "csrf_token": login_csrf,
        },
        follow_redirects=False,
    )
    assert login.status_code == 303, login.status_code
    csrf = _re.search(r'name="csrf_token" value="([^"]+)"', client.get("/app").text).group(1)
    client.post(f"/claims/{claim_id}/interpret", data={"csrf_token": csrf}, follow_redirects=False)

    events = [
        event
        for event in store.audit_trail(claim_id)
        if event["event_type"] == "INTERPRETATION_CITATION_REFUSAL"
    ]
    assert len(events) == 1
    payload = events[0]["payload"]["diagnostic"]
    assert payload == {
        "code": "CITATION_MISMATCH",
        "fact_name": "total_paid",
        "category": "QUOTE_NOT_FOUND",
        "start": 291,
        "end": 299,
    }
    rendered = json.dumps(payload)
    for forbidden in ("USD 5.00", "CANYON", SECRET, "sk-not-real"):
        assert forbidden not in rendered
