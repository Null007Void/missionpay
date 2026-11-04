"""Regressions for the independent review findings on candidate-014.

Every test here maps to one of the four findings, and each is written to fail against the
reviewed code rather than to describe the fixed code:

1. redirects   — the default urllib path followed a 3xx and forwarded ``Authorization``;
2. completion  — a missing/negative/inconsistent usage block, a filtered answer and an
                 absent ``finish_reason`` were all accepted, and nothing stopped a second
                 send under the same authorisation;
3. contract    — the fact vocabulary and citation indexing were never stated to the
                 provider, and the money comparison case-folded before parsing;
4. cost/time   — no pre-send bound, no total deadline, and a token margin that was never
                 measured.

No test contacts a provider. The redirect test uses two loopback servers on ephemeral
ports, which is the only place a real socket is opened.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading

import pytest

from missionpay.ai import (
    FACT_NAMES,
    INTERPRETATION_SCHEMA,
    PROMPT_VERSION,
    ModelInterpretationError,
    interpretation_prompt,
    interpret_receipt_text,
)
from missionpay.config import AiMode, ProviderMode, RuntimeConfig
from missionpay.auth import Role
from tests.catalogue_support import PayloadModel
from missionpay.domain import Activity, Claim, EvidenceIdentity, Money
from missionpay.evidence import EvidenceStore, default_evidence_root
from missionpay.hosted import (
    PINNED_ORIGIN,
    HostedJsonModel,
    HostedModelError,
    UrllibHostedTransport,
)
from missionpay.hosted_qualify import (
    _same_value,
    build_plan,
    compare,
    load_expectation,
    qualify,
)
from missionpay.store import MissionPayStore

FIXTURES = Path(__file__).parent / "fixtures" / "hosted"
RECEIPT = (FIXTURES / "live-receipt.txt").read_text(encoding="utf-8")
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))
DIGEST = EXPECTED["fixture"]["sha256"]
KEY = "synthetic-key-not-a-credential"
MODEL_ID = "gpt-4.1-mini"


# -- scripted transport ------------------------------------------------------


class ScriptedTransport:
    """One scripted answer per call; counts the calls it received."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def request(self, method, url, headers, body, *, max_bytes, timeout):
        self.calls.append({"headers": headers, "body": body, "timeout": timeout})
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response


def _body(status: int, payload: dict) -> tuple[int, bytes]:
    return status, json.dumps(payload).encode("utf-8")


USAGE = {"prompt_tokens": 980, "completion_tokens": 240, "total_tokens": 1220}


def answer(content: str, *, finish: str | None = "stop", usage: dict | None = None):
    payload: dict[str, object] = {
        "choices": [{"message": {"content": content}, "finish_reason": finish}]
    }
    if usage is not False:
        payload["usage"] = USAGE if usage is None else usage
    return _body(200, payload)


def _resolved_interpretation() -> dict:
    """The payload, resolved through the catalogue, in the shape compare() judges.

    compare() judges an interpretation, not a model payload: under v7 the model supplies only
    an excerpt number, and the span exists after MissionPay resolves it.
    """
    from missionpay.ai import interpret_receipt_text

    result = interpret_receipt_text(
        evidence=EvidenceIdentity(DIGEST, "text/plain", "live-receipt.txt"),
        text=RECEIPT,
        model=PayloadModel(good_payload()),
    )
    return {
        "facts": [
            {
                "name": fact.name,
                "value": fact.value,
                "epistemic_status": fact.epistemic_status.value,
                "start": fact.citation.start,
                "end": fact.citation.end,
                "quote": fact.citation.quote,
            }
            for fact in result.facts
        ],
        "missing": list(result.missing),
    }


def good_payload() -> dict:
    """A correct answer in the current schema: one excerpt number per expected fact.

    The number is looked up from the catalogue rather than written down, so this fixture
    cannot drift from the builder. Each expected fact's anchor is carried by exactly one
    excerpt, which is the property the catalogue guarantees.
    """
    from missionpay.catalog import build_catalog

    catalog = build_catalog(RECEIPT)
    facts = []
    for spec in EXPECTED["expected_facts"]:
        matches = [e for e in catalog.excerpts if spec["anchor"] in e.text]
        assert matches, f"no excerpt carries {spec['anchor']!r}"
        entry = min(matches, key=lambda e: (len(e.text), e.excerpt_id))
        facts.append(
            {
                "name": spec["name"],
                "value": spec["value"],
                "epistemic_status": spec["epistemic_status"],
                "excerpt_id": entry.excerpt_id,
            }
        )
    return {"facts": facts, "missing": ["independent proof of attendance"]}


def model(transport=None, **overrides) -> HostedJsonModel:
    kwargs = {
        "base_url": PINNED_ORIGIN,
        "model_id": MODEL_ID,
        "api_key": KEY,
        "max_output_tokens": 512,
        "max_total_tokens": 8192,
        "timeout_seconds": 45,
        "late_response_seconds": 45,
        "transport": transport or ScriptedTransport(answer(json.dumps(good_payload()))),
    }
    kwargs.update(overrides)
    return HostedJsonModel(**kwargs)


# -- a claim to qualify ------------------------------------------------------


def _seed_claim(tmp_path) -> tuple[MissionPayStore, str, str]:
    store = MissionPayStore(str(tmp_path / "qualify.sqlite3"))
    store.initialize()
    store.register_user("v1", "volunteer@example.test", Role.VOLUNTEER, "volunteer-passphrase")
    evidence = EvidenceStore(default_evidence_root(store.path)).save(
        data=RECEIPT.encode("utf-8"), media_type="text/plain", filename="live-receipt.txt"
    )
    session = store.login("volunteer@example.test", "volunteer-passphrase")
    claim = Claim(
        claim_id="claim-live",
        volunteer_id="v1",
        activity=Activity("a1", "RIVER_CLEANUP", date(2026, 11, 3)),
        evidence=EvidenceIdentity(evidence.sha256, "text/plain", "live-receipt.txt"),
        requested=Money("USD", 500),
        recipient="volunteer@example.test",
        expense_category="PROTECTIVE_SUPPLIES",
    )
    store.submit_claim(session.token, claim, revision=1)
    return store, claim.claim_id, evidence.sha256


def _runtime(tmp_path, **overrides) -> RuntimeConfig:
    kwargs = dict(
        provider_mode=ProviderMode.MOCK,
        ai_mode=AiMode.HOSTED,
        hosted_model=MODEL_ID,
        hosted_api_key=KEY,
        hosted_max_output_tokens=512,
        hosted_max_total_tokens=8192,
        hosted_timeout_seconds=45,
        hosted_late_response_seconds=45,
        hosted_confirmed=True,
        database_path=str(tmp_path / "qualify.sqlite3"),
        secure_cookies=False,
    )
    kwargs.update(overrides)
    return RuntimeConfig(**kwargs)


def _config(tmp_path) -> RuntimeConfig:
    return RuntimeConfig(
        provider_mode=ProviderMode.MOCK,
        ai_mode=AiMode.HOSTED,
        hosted_model=MODEL_ID,
        hosted_api_key=KEY,
        hosted_max_output_tokens=512,
        hosted_max_total_tokens=8192,
        hosted_timeout_seconds=45,
        hosted_late_response_seconds=45,
        hosted_confirmed=True,
        database_path=str(tmp_path / "ui.sqlite3"),
        secure_cookies=False,
    )


# -- 1. redirects ------------------------------------------------------------


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_status_is_refused_and_never_followed(status):
    transport = ScriptedTransport((status, b""))
    with pytest.raises(HostedModelError) as excinfo:
        model(transport).complete_json("s", "u")
    assert excinfo.value.code == "HOSTED_REDIRECT_REFUSED"
    assert len(transport.calls) == 1


def _loopback_server(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_the_http_transport_refuses_a_redirect_without_a_second_request():
    """The real transport, two loopback ports, no external network.

    The first server answers 302 with a Location pointing at the second. If redirects
    were followed, the second server would receive the request — and with it the
    Authorization header. Neither happens.
    """
    received: list[tuple[str, str | None]] = []

    class Target(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server naming
            received.append((self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):  # keep the test output quiet
            return

    target, target_thread = _loopback_server(Target)
    target_port = target.server_address[1]

    class Redirector(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server naming
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target_port}/steal")
            self.end_headers()

        def log_message(self, *args):
            return

    redirector, redirector_thread = _loopback_server(Redirector)
    redirector_port = redirector.server_address[1]
    try:
        transport = UrllibHostedTransport(timeout_seconds=5)
        status, body = transport.request(
            "POST",
            f"http://127.0.0.1:{redirector_port}/v1/chat/completions",
            {"Authorization": f"Bearer {KEY}"},
            b"{}",
            max_bytes=1024,
            timeout=5,
        )
    finally:
        for server, thread in ((redirector, redirector_thread), (target, target_thread)):
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    assert status == 302, "the redirect status is returned, not followed"
    assert body == b"", "a redirect body is never read"
    assert received == [], "the redirect target must receive nothing at all"


# -- 2. completion reason, usage and the pre-send bound ----------------------

BAD_USAGE = {
    "missing": None,
    "missing_total": {"prompt_tokens": 980, "completion_tokens": 240},
    "negative": {"prompt_tokens": -1, "completion_tokens": 240, "total_tokens": 239},
    "boolean": {"prompt_tokens": True, "completion_tokens": 240, "total_tokens": 241},
    "float": {"prompt_tokens": 980.5, "completion_tokens": 240, "total_tokens": 1220},
    "string": {"prompt_tokens": "980", "completion_tokens": 240, "total_tokens": 1220},
    "inconsistent": {"prompt_tokens": 980, "completion_tokens": 240, "total_tokens": 999},
    # Above the 8192 budget the v5 payload needs; deliberately not near a boundary.
    "over_budget": {"prompt_tokens": 8000, "completion_tokens": 512, "total_tokens": 8512},
}


@pytest.mark.parametrize(
    "usage,code",
    [
        (None, "HOSTED_USAGE_MISSING"),
        (BAD_USAGE["missing_total"], "HOSTED_USAGE_MISSING"),
        (BAD_USAGE["negative"], "HOSTED_USAGE_INVALID"),
        (BAD_USAGE["boolean"], "HOSTED_USAGE_INVALID"),
        (BAD_USAGE["float"], "HOSTED_USAGE_INVALID"),
        (BAD_USAGE["string"], "HOSTED_USAGE_INVALID"),
        (BAD_USAGE["inconsistent"], "HOSTED_USAGE_INCONSISTENT"),
        (BAD_USAGE["over_budget"], "HOSTED_USAGE_OVER_BUDGET"),
    ],
)
def test_usage_must_be_present_valid_consistent_and_within_budget(usage, code):
    payload = json.loads(json.dumps(good_payload()))
    transport = ScriptedTransport(
        answer(json.dumps(payload), usage=usage if usage is not None else False)
    )
    instance = model(transport)
    with pytest.raises(HostedModelError) as excinfo:
        interpret_receipt_text(
            evidence=EvidenceIdentity(DIGEST, "text/plain", "live-receipt.txt"),
            text=RECEIPT,
            model=instance,
        )
    assert excinfo.value.code == code
    assert len(transport.calls) == 1, "no retry after a rejected answer"
    assert instance.last_usage is None or code == "HOSTED_USAGE_OVER_BUDGET"


@pytest.mark.parametrize(
    "finish,code",
    [
        (None, "HOSTED_INCOMPLETE"),
        ("", "HOSTED_INCOMPLETE"),
        ("tool_calls", "HOSTED_INCOMPLETE"),
        ("content_filter", "HOSTED_CONTENT_FILTERED"),
        ("length", "HOSTED_TRUNCATED"),
    ],
)
def test_a_completion_that_did_not_finish_normally_is_refused(finish, code):
    transport = ScriptedTransport(answer(json.dumps(good_payload()), finish=finish))
    with pytest.raises(HostedModelError) as excinfo:
        model(transport).complete_json("s", "u")
    assert excinfo.value.code == code
    assert len(transport.calls) == 1


def test_valid_usage_is_retained_for_evidence():
    instance = model()
    instance.complete_json("s", "u")
    assert instance.last_usage == USAGE


def test_the_usage_is_written_to_the_audit_trail(tmp_path):
    store, claim_id, _digest = _seed_claim(tmp_path)
    store.record_interpretation_usage(claim_id, 1, USAGE, model_id=f"openai:{MODEL_ID}")
    events = [e for e in store.audit_trail(claim_id) if e["event_type"] == "INTERPRETATION_USAGE"]
    assert len(events) == 1
    assert events[0]["payload"]["usage"] == USAGE
    assert events[0]["payload"]["model_id"] == f"openai:{MODEL_ID}"


def test_an_over_budget_request_is_refused_before_any_socket_is_opened():
    """A post-response check cannot undo spend, so the bound is applied first."""
    transport = ScriptedTransport(answer(json.dumps(good_payload())))
    instance = model(transport, max_total_tokens=600)
    with pytest.raises(HostedModelError) as excinfo:
        instance.complete_json(*interpretation_prompt(RECEIPT))
    assert excinfo.value.code == "HOSTED_REQUEST_OVER_BUDGET"
    assert transport.calls == [], "the request was refused before it left"


class _Clock:
    """First read is the send time; every later read is 46 seconds on."""

    def __init__(self):
        self.reads = 0

    def __call__(self):
        self.reads += 1
        return 100.0 if self.reads == 1 else 146.0


def test_a_late_response_is_refused_after_the_fact(monkeypatch):
    monkeypatch.setattr("missionpay.hosted.time.monotonic", _Clock())
    transport = ScriptedTransport(answer(json.dumps(good_payload())))
    with pytest.raises(HostedModelError) as excinfo:
        model(transport, late_response_seconds=45).complete_json("s", "u")
    assert excinfo.value.code == "HOSTED_LATE_RESPONSE"
    assert len(transport.calls) == 1, "an answer after the deadline is terminal"


def test_the_socket_timeout_and_the_deadline_are_reported_separately():
    described = model(timeout_seconds=30, late_response_seconds=45).describe()
    assert described["timeout_seconds"] == 30
    assert described["late_response_seconds"] == 45
    transport = ScriptedTransport(answer(json.dumps(good_payload())))
    model(transport, timeout_seconds=30, late_response_seconds=45).complete_json("s", "u")
    assert transport.calls[0]["timeout"] == 30, "per-operation socket timeout"


def test_the_published_token_margin_is_an_estimate_with_headroom():
    instance = model()
    estimate = instance.preview(*interpretation_prompt(RECEIPT))
    assert estimate["estimated_total_tokens"] <= instance.max_total_tokens
    headroom = 1 - estimate["estimated_total_tokens"] / instance.max_total_tokens
    assert headroom >= 0.2, f"only {headroom:.0%} headroom above the estimate"


# -- 3. prompt, contract and frozen verdict ---------------------------------


def test_the_prompt_and_the_schema_state_the_vocabulary_and_citation_mechanism():
    """Under v7 the mechanism is a catalogue number, not offsets into the text."""
    system, _user = interpretation_prompt(RECEIPT)
    for name in FACT_NAMES:
        assert name in system, name
    assert "excerpt_id" in system
    assert "catalogue" in system.lower()
    fact_schema = INTERPRETATION_SCHEMA["properties"]["facts"]["items"]
    assert fact_schema["properties"]["name"]["enum"] == list(FACT_NAMES)
    assert "excerpt_id" in fact_schema["required"]
    assert "catalogue" in fact_schema["properties"]["excerpt_id"]["description"].lower()


def test_the_prompt_does_not_contain_the_expected_answers():
    """Naming the facts is the contract; stating their values would be the answer."""
    system, user = interpretation_prompt(RECEIPT)
    for leak in ("CANYON", "4412", "Moab", "river cleanup"):
        assert leak.lower() not in system.lower(), leak
    assert user.startswith("<receipt-text>\n" + RECEIPT + "\n</receipt-text>")
    assert "<excerpt-catalogue>" in user


def test_a_fact_name_outside_the_contract_is_refused():
    payload = good_payload()
    payload["facts"][0]["name"] = "date"
    transport = ScriptedTransport(answer(json.dumps(payload)))
    with pytest.raises(ModelInterpretationError) as excinfo:
        interpret_receipt_text(
            evidence=EvidenceIdentity(DIGEST, "text/plain", "live-receipt.txt"),
            text=RECEIPT,
            model=model(transport),
        )
    assert str(excinfo.value) == "MODEL_FACT_NAME_UNKNOWN"


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("USD 5.00", "5.00", True),
        ("$5.00", "USD 5.00", True),
        ("5.00 USD", "5.00", True),
        ("5", "5.00", True),
        ("USD 5.00", "EUR 5.00", False),
        ("USD 5.00", "50.00", False),
        ("5.00", "5.01", False),
        ("5.00", "5.000", False),
        ("2", "2.00", True),
        ("2", "twenty", False),
        ("  Safety   Whistle ", "safety whistle", True),
    ],
)
def test_money_is_parsed_before_case_folding(left, right, expected):
    assert _same_value(left, right) is expected


def test_money_comparison_uses_decimals_not_floats():
    # 0.1 + 0.2 != 0.3 in binary floating point; Decimal is exact.
    assert _same_value("0.30", "0.1") is False
    assert _same_value("0.30", "0.3") is True
    assert Decimal("0.30") == Decimal("0.3")


def test_the_frozen_expectation_passes_a_correct_answer_and_fails_a_wrong_one():
    expectation = load_expectation(FIXTURES)
    before = (FIXTURES / "expected.json").read_bytes()

    good = compare(_resolved_interpretation(), expectation, RECEIPT)
    assert good["passed"], good["mismatches"]
    assert good["unexpected_facts"] == []

    wrong_amount = _resolved_interpretation()
    for fact in wrong_amount["facts"]:
        if fact["name"] == "total_paid":
            fact["value"] = "USD 50.00"
    verdict = compare(wrong_amount, expectation, RECEIPT)
    assert not verdict["passed"]
    assert any("total_paid" in item for item in verdict["mismatches"])

    # A citation that no longer matches the receipt text: the span and the quote disagree.
    paraphrased = _resolved_interpretation()
    for fact in paraphrased["facts"]:
        if fact["name"] == "merchant":
            fact["quote"] = "Canyon Supply"
    verdict = compare(paraphrased, expectation, RECEIPT)
    assert not verdict["passed"]
    assert any("merchant" in item for item in verdict["mismatches"])

    assert (FIXTURES / "expected.json").read_bytes() == before, "the verdict never rewrites the expectation"


def test_an_answer_that_claims_nothing_is_missing_is_refused():
    expectation = load_expectation(FIXTURES)
    interpretation = _resolved_interpretation()
    interpretation["missing"] = []
    verdict = compare(interpretation, expectation, RECEIPT)
    assert not verdict["passed"]
    assert any("missing" in item for item in verdict["mismatches"])


# -- 4. one-shot qualification ---------------------------------------------


class _ProbingTransport:
    """Records what the database said *during* the request."""

    def __init__(self, store, authorization_id, response):
        self.store = store
        self.authorization_id = authorization_id
        self.response = response
        self.calls = 0
        self.state_at_send: list[str | None] = []

    def request(self, method, url, headers, body, *, max_bytes, timeout):
        self.calls += 1
        row = self.store.hosted_attempt(self.authorization_id)
        self.state_at_send.append(None if row is None else str(row["state"]))
        return self.response


def test_the_attempt_is_recorded_before_the_request_leaves(tmp_path):
    store, claim_id, digest = _seed_claim(tmp_path)
    transport = _ProbingTransport(store, "auth-1", answer(json.dumps(good_payload())))
    outcome = qualify(
        store,
        _runtime(tmp_path),
        authorization_id="auth-1",
        claim_id=claim_id,
        expectation=EXPECTED,
        model=model(transport=transport),
    )
    assert outcome["status"] == "QUALIFIED"
    assert transport.calls == 1
    assert transport.state_at_send == ["RECORDED"], "the row must exist before the send"
    row = store.hosted_attempt("auth-1")
    assert row["state"] == "SUCCEEDED"
    assert row["usage"] == USAGE
    assert row["evidence_sha256"] == digest
    assert row["model_id"] == f"openai:{MODEL_ID}"


def test_a_second_attempt_under_the_same_authorisation_is_refused(tmp_path):
    store, claim_id, _digest = _seed_claim(tmp_path)
    transport = ScriptedTransport(answer(json.dumps(good_payload())))
    instance = model(transport)
    first = qualify(store, _runtime(tmp_path), authorization_id="auth-1", claim_id=claim_id, expectation=EXPECTED, model=instance)
    assert first["status"] == "QUALIFIED"

    second = qualify(store, _runtime(tmp_path), authorization_id="auth-1", claim_id=claim_id, expectation=EXPECTED, model=instance)
    assert second["status"] == "ALREADY_ATTEMPTED"
    assert second["sent"] is False
    assert len(transport.calls) == 1, "the second run must not reach the transport"


def test_the_refusal_survives_a_restart(tmp_path):
    store, claim_id, _digest = _seed_claim(tmp_path)
    transport = ScriptedTransport(answer(json.dumps(good_payload())))
    qualify(store, _runtime(tmp_path), authorization_id="auth-1", claim_id=claim_id, expectation=EXPECTED, model=model(transport))

    reopened = MissionPayStore(str(tmp_path / "qualify.sqlite3"))
    reopened.initialize()
    again = qualify(
        reopened, _runtime(tmp_path), authorization_id="auth-1", claim_id=claim_id, expectation=EXPECTED, model=model(transport)
    )
    assert again["status"] == "ALREADY_ATTEMPTED"
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "response,state,code",
    [
        (_body(401, {"error": "nope"}), "FAILED", "HOSTED_AUTH_FAILED"),
        (_body(429, {}), "FAILED", "HOSTED_RATE_LIMITED"),
        (TimeoutError("timed out"), "TIMEOUT", "HOSTED_TIMEOUT"),
        (ConnectionResetError("reset"), "UNCERTAIN", "HOSTED_NETWORK_ERROR"),
    ],
)
def test_a_failed_attempt_consumes_the_authorisation(tmp_path, response, state, code):
    store, claim_id, _digest = _seed_claim(tmp_path)
    transport = ScriptedTransport(response)
    instance = model(transport)
    outcome = qualify(store, _runtime(tmp_path), authorization_id="auth-1", claim_id=claim_id, expectation=EXPECTED, model=instance)
    assert outcome["status"] == state
    assert outcome["code"] == code
    assert store.hosted_attempt("auth-1")["state"] == state

    again = qualify(store, _runtime(tmp_path), authorization_id="auth-1", claim_id=claim_id, expectation=EXPECTED, model=instance)
    assert again["status"] == "ALREADY_ATTEMPTED"
    assert len(transport.calls) == 1, "a failure is never retried automatically"


def test_a_new_authorisation_allows_a_new_attempt(tmp_path):
    """The guarantee is one request per authorisation, not a permanently single-use app."""
    store, claim_id, _digest = _seed_claim(tmp_path)
    transport = ScriptedTransport(
        answer(json.dumps(good_payload())), answer(json.dumps(good_payload()))
    )
    instance = model(transport)
    first = qualify(store, _runtime(tmp_path), authorization_id="auth-1", claim_id=claim_id, expectation=EXPECTED, model=instance)
    second = qualify(store, _runtime(tmp_path), authorization_id="auth-2", claim_id=claim_id, expectation=EXPECTED, model=instance)
    assert first["status"] == "QUALIFIED"
    assert second["status"] == "QUALIFIED"
    assert len(transport.calls) == 2
    assert {row["authorization_id"] for row in store.hosted_attempts()} == {"auth-1", "auth-2"}


def test_a_dry_run_records_nothing(tmp_path):
    store, claim_id, _digest = _seed_claim(tmp_path)
    plan, text, evidence = build_plan(
        store,
        _runtime(tmp_path),
        authorization_id="auth-1",
        claim_id=claim_id,
        revision=None,
        expectation=EXPECTED,
        model=model(),
    )
    assert store.hosted_attempts() == []
    assert plan.estimate["budget_tokens"] == 8192
    assert plan.as_dict()["estimate_is_a_measurement"] is False
    assert plan.expectation_sha256 == __import__("hashlib").sha256(
        json.dumps(EXPECTED, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert plan.late_response_seconds == 45 and plan.timeout_seconds == 45
    assert text == RECEIPT
    assert evidence.sha256 == DIGEST


def test_an_over_budget_plan_is_refused_without_consuming_the_authorisation(tmp_path):
    from missionpay.hosted_qualify import QualificationRefused

    store, claim_id, _digest = _seed_claim(tmp_path)
    with pytest.raises(QualificationRefused) as excinfo:
        qualify(
            store,
            _runtime(tmp_path),
            authorization_id="auth-1",
            claim_id=claim_id,
            expectation=EXPECTED,
            model=model(max_total_tokens=700),
        )
    assert excinfo.value.code == "HOSTED_REQUEST_OVER_BUDGET"
    assert store.hosted_attempts() == [], "nothing was sent, so nothing is consumed"


def test_the_pinned_columns_cannot_be_rewritten(tmp_path):
    store, claim_id, _digest = _seed_claim(tmp_path)
    transport = ScriptedTransport(answer(json.dumps(good_payload())))
    qualify(store, _runtime(tmp_path), authorization_id="auth-1", claim_id=claim_id, expectation=EXPECTED, model=model(transport))

    connection = sqlite3.connect(str(tmp_path / "qualify.sqlite3"))
    try:
        with pytest.raises(sqlite3.IntegrityError) as excinfo:
            connection.execute(
                "UPDATE hosted_attempts SET model_id='other-model' WHERE authorization_id=?",
                ("auth-1",),
            )
        assert "IMMUTABLE_HOSTED_ATTEMPT_PINNING" in str(excinfo.value)
        with pytest.raises(sqlite3.IntegrityError) as excinfo:
            connection.execute("DELETE FROM hosted_attempts WHERE authorization_id=?", ("auth-1",))
        assert "IMMUTABLE_HOSTED_ATTEMPT" in str(excinfo.value)
    finally:
        connection.close()


def test_the_web_route_does_not_use_the_qualification_journal(tmp_path):
    """The product stays multi-use: only the paid qualification command is one-shot."""
    from fastapi.testclient import TestClient

    from missionpay import web as web_module

    store, claim_id, _digest = _seed_claim(tmp_path)
    app = web_module.create_app(store, config=_config(tmp_path))
    client = TestClient(app)
    client.post(
        "/login",
        data={"email": "volunteer@example.test", "password": "volunteer-passphrase"},
        follow_redirects=False,
    )
    page = client.get("/app").text
    import re as _re

    csrf = _re.search(r'name="csrf_token" value="([^"]+)"', page).group(1)
    client.post(f"/claims/{claim_id}/interpret", data={"csrf_token": csrf}, follow_redirects=False)
    assert store.hosted_attempts() == []
