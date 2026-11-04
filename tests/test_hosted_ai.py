"""Hosted (structured-output) adapter: gates, failure taxonomy, and the full workflow.

The adapter is the only place in this product that can spend money or send a donor's
receipt to a third party, so the tests are organised around what must be impossible:
construct with a bad origin, succeed without a key, retry, fall back, leak the key into
an error, accept a truncated or refused answer, and — the subtle one — serve a cached
interpretation produced by a different provider or model.

Every expectation about the $5 receipt was fixed in ``fixtures/hosted/expected.json``
before this file was written. Nothing here edits an expected fact or span to match
model output; the scripted responses are built *from* the expectation.
"""

from __future__ import annotations

import json
from pathlib import Path
import os
import re
import tempfile

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient

import missionpay.web as web_module
from missionpay import hosted as hosted_module
from missionpay.ai import (
    EpistemicStatus,
    ModelInterpretationError,
    interpret_receipt_text,
)
from missionpay.auth import Role
from missionpay.bootstrap import bootstrap_demo_store
from missionpay.config import AiMode, ConfigurationError, ProviderMode, RuntimeConfig, from_environment
from missionpay.domain import EvidenceIdentity
from missionpay.hosted import HostedJsonModel, HostedModelError
from missionpay.store import MissionPayStore
from missionpay.web import create_app

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hosted"
RECEIPT = (FIXTURES / "live-receipt.txt").read_text(encoding="utf-8")
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))
EVIDENCE = EvidenceIdentity(EXPECTED["fixture"]["sha256"], "text/plain", "live-receipt.txt")

#: A value that is obviously not a credential and must never appear in a failure.
KEY = "sk-test-not-a-credential-000000000000"


# -- scripted transport ------------------------------------------------------


class ScriptedTransport:
    """One scripted answer per call; records what it was asked for."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def request(self, method, url, headers, body, *, max_bytes, timeout):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "body": body,
                "max_bytes": max_bytes,
                "timeout": timeout,
            }
        )
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response


def _json_body(status: int, payload: dict) -> tuple[int, bytes]:
    return status, json.dumps(payload).encode("utf-8")


def ok(content: str, finish: str = "stop") -> tuple[int, bytes]:
    return _json_body(
        200,
        {
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 640, "completion_tokens": 240, "total_tokens": 880},
        },
    )


def refused() -> tuple[int, bytes]:
    return _json_body(
        200,
        {
            "choices": [
                {"message": {"content": None, "refusal": "cannot assist"}, "finish_reason": "stop"}
            ]
        },
    )


# -- the expected interpretation, as a model response -----------------------


def expected_facts_payload() -> dict:
    """A correct answer in the v7 schema: the catalogue number that carries each anchor."""
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


def expected_json() -> str:
    return json.dumps(expected_facts_payload())


# The comparison rule lives in the product, so the verdict and this assertion cannot
# drift apart. It parses money before folding case and compares decimals.
from missionpay.hosted_qualify import _same_value


def assert_matches_expectation(result) -> None:
    by_name = {fact.name: fact for fact in result.facts}
    assert set(by_name) == {fact["name"] for fact in EXPECTED["expected_facts"]}
    for expected in EXPECTED["expected_facts"]:
        fact = by_name[expected["name"]]
        assert fact.epistemic_status.value == expected["epistemic_status"], expected["name"]
        assert _same_value(expected["value"], fact.value), (expected["name"], fact.value)
        assert RECEIPT[fact.citation.start : fact.citation.end] == fact.citation.quote
        # The excerpt is a whole line, so it contains the anchor rather than equalling it.
        assert expected["anchor"] in fact.citation.quote, expected["name"]
        assert fact.citation.excerpt_id is not None
    assert result.missing and all(isinstance(item, str) and item for item in result.missing)


# -- config gating -----------------------------------------------------------


def _hosted_env(**overrides) -> dict[str, str]:
    # A hosted configuration reads the credential file even when the key comes from the
    # environment, so a test that does not pin the file reads whatever the developer has
    # saved on this machine. Pinning it to a path that cannot exist keeps the suite
    # hermetic and keeps a real key out of every test process.
    env = {
        "MISSIONPAY_AI_MODE": "hosted",
        "MISSIONPAY_HOSTED_AI_CONFIRM": "1",
        "MISSIONPAY_HOSTED_API_KEY": KEY,
        "MISSIONPAY_HOSTED_MODEL": "gpt-4.1-mini",
        "MISSIONPAY_HOSTED_MAX_OUTPUT_TOKENS": "512",
        "MISSIONPAY_HOSTED_MAX_TOTAL_TOKENS": "8192",
    }
    env.update(overrides)
    return {key: value for key, value in env.items() if value is not None}


def test_a_complete_hosted_configuration_is_accepted():
    config = from_environment(_hosted_env())
    assert config.ai_mode is AiMode.HOSTED
    assert config.hosted_model == "gpt-4.1-mini"
    assert config.hosted_max_output_tokens == 512
    described = config.describe()
    assert described["hosted_api_key_present"] is True
    assert described["hosted_model"] == "gpt-4.1-mini"
    assert KEY not in json.dumps(described)


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"MISSIONPAY_HOSTED_AI_CONFIRM": None}, "HOSTED_AI_NOT_CONFIRMED"),
        ({"MISSIONPAY_HOSTED_API_KEY": None}, "HOSTED_AI_KEY_REQUIRED"),
        ({"MISSIONPAY_HOSTED_MODEL": None}, "HOSTED_MODEL_REQUIRED"),
        ({"MISSIONPAY_HOSTED_BASE_URL": "https://api.openai.com.attacker.example"}, "HOSTED_ORIGIN_REQUIRED"),
        ({"MISSIONPAY_HOSTED_BASE_URL": "https://api.openai.com/v1"}, "HOSTED_ORIGIN_REQUIRED"),
        ({"MISSIONPAY_HOSTED_MAX_OUTPUT_TOKENS": None}, "HOSTED_MAX_OUTPUT_TOKENS_REQUIRED"),
        ({"MISSIONPAY_HOSTED_MAX_OUTPUT_TOKENS": "5000"}, "HOSTED_MAX_OUTPUT_TOKENS_INVALID"),
        ({"MISSIONPAY_HOSTED_TIMEOUT_SECONDS": "600"}, "HOSTED_TIMEOUT_INVALID"),
    ],
)
def test_hosted_mode_fails_closed(overrides, code):
    with pytest.raises(ConfigurationError) as excinfo:
        from_environment(_hosted_env(**overrides))
    assert str(excinfo.value) == code


def test_hosted_mode_is_unreachable_without_the_second_key():
    """A mode typo alone must not start spending money."""
    env = _hosted_env(MISSIONPAY_HOSTED_AI_CONFIRM=None)
    with pytest.raises(ConfigurationError) as excinfo:
        from_environment(env)
    assert str(excinfo.value) == "HOSTED_AI_NOT_CONFIRMED"


def test_the_default_configuration_stays_offline():
    config = from_environment({"MISSIONPAY_PROVIDER_MODE": "mock"})
    assert config.ai_mode is AiMode.FIXTURE
    assert config.hosted_api_key is None


# -- adapter construction ----------------------------------------------------


def _model(transport=None, **overrides):
    kwargs = {
        "base_url": hosted_module.PINNED_ORIGIN,
        "model_id": "gpt-4.1-mini",
        "api_key": KEY,
        "max_output_tokens": 512,
        "max_total_tokens": 8192,
        "timeout_seconds": 45,
        "late_response_seconds": 45,
        "transport": transport or ScriptedTransport(ok(expected_json())),
    }
    kwargs.update(overrides)
    return HostedJsonModel(**kwargs)


@pytest.mark.parametrize(
    "overrides,code",
    [
        ({"api_key": ""}, "HOSTED_AI_KEY_REQUIRED"),
        ({"model_id": ""}, "HOSTED_MODEL_REQUIRED"),
        ({"base_url": "https://api.openai.com.attacker.example"}, "HOSTED_ORIGIN_REQUIRED"),
        ({"base_url": "http://api.openai.com"}, "HOSTED_ORIGIN_REQUIRED"),
        ({"max_output_tokens": 0}, "HOSTED_MAX_OUTPUT_TOKENS_INVALID"),
        ({"max_output_tokens": 4097}, "HOSTED_MAX_OUTPUT_TOKENS_INVALID"),
        ({"timeout_seconds": 0}, "HOSTED_TIMEOUT_INVALID"),
        ({"timeout_seconds": 121}, "HOSTED_TIMEOUT_INVALID"),
    ],
)
def test_adapter_refuses_unsafe_or_unbounded_construction(overrides, code):
    with pytest.raises(HostedModelError) as excinfo:
        _model(**overrides)
    assert excinfo.value.code == code


def test_the_key_is_absent_from_repr_describe_and_errors():
    model = _model()
    assert KEY not in repr(model)
    assert KEY not in json.dumps(model.describe())
    assert model.describe()["api_key_present"] is True
    failing = _model(ScriptedTransport(_json_body(401, {"error": "nope"})))
    with pytest.raises(HostedModelError) as excinfo:
        failing.complete_json("s", "u")
    assert KEY not in str(excinfo.value)
    assert KEY not in repr(excinfo.value)


# -- the request -------------------------------------------------------------


def test_one_request_with_a_pinned_origin_and_a_strict_schema():
    transport = ScriptedTransport(ok(expected_json()))
    model = _model(transport)
    content = model.complete_json("system", "user")

    assert content == expected_json()
    assert len(transport.calls) == 1, "a single attempt, never a retry"
    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://api.openai.com/v1/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"

    body = json.loads(call["body"])
    assert body["model"] == "gpt-4.1-mini"
    assert body["temperature"] == 0
    assert body["max_tokens"] == 512
    assert body["store"] is False
    assert body["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    schema = body["response_format"]
    assert schema["type"] == "json_schema"
    assert schema["json_schema"]["strict"] is True
    assert schema["json_schema"]["schema"]["additionalProperties"] is False


def test_usage_is_recorded_without_the_response_body():
    model = _model()
    model.complete_json("s", "u")
    assert model.last_usage == {"prompt_tokens": 640, "completion_tokens": 240, "total_tokens": 880}


# -- failure taxonomy --------------------------------------------------------


@pytest.mark.parametrize(
    "response,code",
    [
        (_json_body(401, {}), "HOSTED_AUTH_FAILED"),
        (_json_body(403, {}), "HOSTED_AUTH_FAILED"),
        (_json_body(429, {}), "HOSTED_RATE_LIMITED"),
        (_json_body(400, {}), "HOSTED_REQUEST_REJECTED"),
        (_json_body(404, {}), "HOSTED_MODEL_UNAVAILABLE"),
        (_json_body(500, {}), "HOSTED_PROVIDER_ERROR"),
        (_json_body(503, {}), "HOSTED_PROVIDER_ERROR"),
        (_json_body(418, {}), "HOSTED_HTTP_418"),
        ((200, b"not json at all"), "HOSTED_RESPONSE_INVALID"),
        (_json_body(200, {"choices": []}), "HOSTED_RESPONSE_INVALID"),
        (_json_body(200, {"choices": [{"message": "nope"}]}), "HOSTED_RESPONSE_INVALID"),
        (refused(), "HOSTED_REFUSED"),
        (ok('{"facts": [{"na', finish="length"), "HOSTED_TRUNCATED"),
        (ok("   "), "HOSTED_CONTENT_MISSING"),
        (
            _json_body(
                200,
                {
                    "choices": [{"message": {"content": None}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
                },
            ),
            "HOSTED_CONTENT_MISSING",
        ),
    ],
)
def test_every_failure_is_typed_and_never_retried(response, code):
    transport = ScriptedTransport(response)
    model = _model(transport)
    with pytest.raises(HostedModelError) as excinfo:
        model.complete_json("s", "u")
    assert excinfo.value.code == code
    assert len(transport.calls) == 1, "a failure must not be retried"


def test_a_transport_failure_is_typed_and_not_retried():
    transport = ScriptedTransport(TimeoutError("timed out"))
    with pytest.raises(HostedModelError) as excinfo:
        _model(transport).complete_json("s", "u")
    assert excinfo.value.code == "HOSTED_TIMEOUT"
    assert len(transport.calls) == 1


def test_an_oversized_response_is_refused_without_parsing(monkeypatch):
    big = b"{" + b" " * (hosted_module.MAX_RESPONSE_BYTES + 10) + b"}"
    transport = ScriptedTransport((200, big))
    with pytest.raises(HostedModelError) as excinfo:
        _model(transport).complete_json("s", "u")
    assert excinfo.value.code == "HOSTED_RESPONSE_TOO_LARGE"


def test_the_urlopen_transport_bounds_the_read():
    """The real transport enforces the ceiling, not only the fake one."""
    transport = hosted_module.UrllibHostedTransport(timeout_seconds=5)
    assert transport.timeout_seconds == 5


# -- the validator still decides --------------------------------------------


def _interpret_with(content: str):
    transport = ScriptedTransport(ok(content))
    return interpret_receipt_text(evidence=EVIDENCE, text=RECEIPT, model=_model(transport))


def test_a_complete_interpretation_is_accepted_and_matches_the_expectation():
    result = _interpret_with(expected_json())
    assert result.model_id == "gpt-4.1-mini"
    assert result.evidence_sha256 == EXPECTED["fixture"]["sha256"]
    assert_matches_expectation(result)


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda payload: {"facts": "not a list", "missing": []}, "MODEL_SCHEMA_INVALID"),
        (lambda payload: {"facts": [], "missing": "nope"}, "MODEL_SCHEMA_INVALID"),
        (lambda payload: {**payload, "facts": [{**payload["facts"][0], "name": ""}]}, "MODEL_SCHEMA_INVALID"),
        (lambda payload: {**payload, "facts": [{**payload["facts"][0], "epistemic_status": "guessed"}]}, "MODEL_SCHEMA_INVALID"),
        # A malformed or shifted span is no longer a refusal: since candidate-019 the quote
        # decides, and these are covered as resolutions in test_citation_resolution.py.
        # An identifier the catalogue never offered is a refusal; text-shaped keys are now a
        # schema violation, because the model must not return them at all.
        (lambda payload: {**payload, "facts": [{**payload["facts"][0], "excerpt_id": 9999}]}, "CITATION_MISMATCH"),
        (lambda payload: {**payload, "facts": [{**payload["facts"][0], "excerpt_id": 0}]}, "CITATION_MISMATCH"),
        (lambda payload: {**payload, "facts": [{**payload["facts"][0], "quote": "something else"}]}, "MODEL_SCHEMA_INVALID"),
        # The advertised schema forbids additional properties; an answer carrying one
        # is not silently trimmed.
        (lambda payload: {**payload, "confidence": 0.9}, "MODEL_SCHEMA_INVALID"),
        (lambda payload: {**payload, "facts": [{**payload["facts"][0], "confidence": 0.9}]}, "MODEL_SCHEMA_INVALID"),
    ],
)
def test_malformed_and_unsupported_answers_are_refused(mutate, code):
    payload = mutate(expected_facts_payload())
    with pytest.raises(ModelInterpretationError) as excinfo:
        _interpret_with(json.dumps(payload))
    assert str(excinfo.value) == code


def test_unparseable_json_is_refused():
    with pytest.raises(ModelInterpretationError) as excinfo:
        _interpret_with('{"facts": [')
    assert str(excinfo.value) == "MODEL_JSON_INVALID"


def test_a_truncated_answer_never_reaches_the_validator():
    transport = ScriptedTransport(ok('{"facts": [{"na', finish="length"))
    with pytest.raises(HostedModelError) as excinfo:
        interpret_receipt_text(evidence=EVIDENCE, text=RECEIPT, model=_model(transport))
    assert excinfo.value.code == "HOSTED_TRUNCATED"


def test_the_span_always_comes_from_the_catalogue_never_from_the_answer():
    """There is no model arithmetic left to shift: MissionPay resolves the number."""
    result = _interpret_with(json.dumps(expected_facts_payload()))
    assert [fact.offsets_derived for fact in result.facts] == [False] * 8
    assert_matches_expectation(result)
    assert result.facts[0].citation.start == EXPECTED["expected_facts"][0]["canonical_start"]
    # And the citation is the catalogue's text, not anything the answer typed.
    assert result.facts[0].citation.quote in RECEIPT


def test_an_answer_that_still_returns_a_quote_is_refused_by_the_schema():
    """The model must not transcribe. Text-shaped keys are a schema violation, not a citation."""
    payload = expected_facts_payload()
    payload["facts"][0]["quote"] = "Canyon Supply Cooperative"
    with pytest.raises(ModelInterpretationError) as excinfo:
        _interpret_with(json.dumps(payload))
    assert str(excinfo.value) == "MODEL_SCHEMA_INVALID"


# -- the workflow, through the application ----------------------------------


def _app(tmp_path, config, store=None):
    if store is None:
        store = MissionPayStore(tmp_path / "ui.sqlite3", password_hasher=PasswordHasher(time_cost=1))
        store.initialize()
        bootstrap_demo_store(store.path)
    return store, TestClient(create_app(store, config=config), base_url="http://testserver")


def _hosted_config(tmp_path, model="gpt-4.1-mini"):
    return RuntimeConfig(
        provider_mode=ProviderMode.MOCK,
        ai_mode=AiMode.HOSTED,
        hosted_model=model,
        hosted_api_key=KEY,
        hosted_max_output_tokens=512,
        hosted_max_total_tokens=8192,
        hosted_timeout_seconds=45,
        hosted_late_response_seconds=45,
        hosted_confirmed=True,
        database_path=str(tmp_path / "ui.sqlite3"),
        secure_cookies=False,
    )


def _csrf(client, path="/app"):
    return re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).text).group(1)


def _volunteer_submits(client, path="/claims/demo"):
    client.post(
        "/login",
        data={
            "email": "volunteer@missionpay.local",
            "password": "volunteer-demo-passphrase",
            "csrf_token": _csrf(client, "/login"),
        },
        follow_redirects=False,
    )
    if path == "/claims/demo":
        submitted = client.post("/claims/demo", data={"csrf_token": _csrf(client)}, follow_redirects=False)
        return submitted.headers["location"]
    uploaded = client.post(
        "/claims/upload",
        data={
            "csrf_token": _csrf(client),
            "amount_cents": "500",
            "activity_code": "RIVER_CLEANUP",
            "expense_category": "PROTECTIVE_SUPPLIES",
        },
        files={"evidence": ("live-receipt.txt", RECEIPT.encode("utf-8"), "text/plain")},
        follow_redirects=False,
    )
    return uploaded.headers["location"]


def _inject(monkeypatch, transport):
    real = hosted_module.HostedJsonModel
    monkeypatch.setattr(
        web_module, "HostedJsonModel", lambda **kwargs: real(**{**kwargs, "transport": transport})
    )


def test_a_provider_failure_is_recorded_as_a_refusal_not_a_crash(tmp_path, monkeypatch):
    """A 401 must leave the claim uninterpreted with a named code, not a 500."""
    transport = ScriptedTransport(_json_body(401, {"error": "invalid key"}))
    _inject(monkeypatch, transport)
    _store_, client = _app(tmp_path, _hosted_config(tmp_path))
    path = _volunteer_submits(client, "/claims/upload")

    response = client.post(
        f"{path}/interpret", data={"csrf_token": _csrf(client)}, follow_redirects=False
    )
    assert response.status_code == 303, "a provider failure must not become a 500"

    page = client.get(path).text
    assert "Interpretation refused: HOSTED_AUTH_FAILED" in page
    assert "SIMULATED INTERPRETATION FIXTURE" not in page
    assert "No payment intent" in page
    assert len(transport.calls) == 1, "one attempt; no retry after an auth failure"


def test_the_complete_hosted_journey_on_the_five_dollar_receipt(tmp_path, monkeypatch):
    transport = ScriptedTransport(ok(expected_json()))
    _inject(monkeypatch, transport)
    _store_, client = _app(tmp_path, _hosted_config(tmp_path))
    path = _volunteer_submits(client, "/claims/upload")
    assert path.startswith("/claims/claim-")

    page = client.get(path).text
    assert "$5.00" in page
    assert "NOT RUN — THE AI STAGE IS INCOMPLETE" in page

    client.post(f"{path}/interpret", data={"csrf_token": _csrf(client)})
    assert len(transport.calls) == 1

    page = client.get(path).text
    assert "LIVE HOSTED MODEL — openai:gpt-4.1-mini" in page
    assert "SIMULATED INTERPRETATION FIXTURE" not in page, "a hosted run must not carry demo copy"
    assert "CANYON SUPPLY COOPERATIVE" in page
    assert "USD 5.00" in page
    assert "Observed fact" in page and "Citation" in page
    assert "No payment intent" in page, "interpretation alone must not create authority"

    # The reviewer approves exactly the interpretation the page displayed.
    reviewer = TestClient(create_app(
        MissionPayStore(tmp_path / "ui.sqlite3", password_hasher=PasswordHasher(time_cost=1)),
        config=_hosted_config(tmp_path),
    ), base_url="http://testserver")
    reviewer.post(
        "/login",
        data={
            "email": "reviewer@missionpay.local",
            "password": "reviewer-demo-passphrase",
            "csrf_token": _csrf(reviewer, "/login"),
        },
        follow_redirects=False,
    )
    page = reviewer.get(path).text
    identity = {
        name: re.search(rf'name="{name}"\s+value="([^"]+)"', page, re.S).group(1)
        for name in ("interpretation_id", "interpretation_output_sha256")
    }
    approved = reviewer.post(
        f"{path}/approve",
        data={
            "csrf_token": _csrf(reviewer),
            "revision": "1",
            "amount_cents": "500",
            **identity,
        },
        follow_redirects=False,
    )
    assert approved.status_code == 303
    approved_page = reviewer.get(path).text
    assert "Exact authority frozen" in approved_page
    assert "Not attempted" in approved_page
    # Payment is still refused: a hosted interpretation is not authority to pay.
    paid = reviewer.post(f"{path}/pay", data={"csrf_token": _csrf(reviewer)}, follow_redirects=False)
    assert paid.status_code == 409
    assert "MOCK_MODE_REFUSES_PROVIDER_OPERATIONS" in paid.text


def test_an_identical_configuration_reuses_without_a_second_request(tmp_path, monkeypatch):
    transport = ScriptedTransport(ok(expected_json()))
    _inject(monkeypatch, transport)
    _store_, client = _app(tmp_path, _hosted_config(tmp_path))
    path = _volunteer_submits(client, "/claims/upload")

    client.post(f"{path}/interpret", data={"csrf_token": _csrf(client)})
    client.post(f"{path}/interpret", data={"csrf_token": _csrf(client)})
    assert len(transport.calls) == 1


def test_changing_the_model_does_not_serve_the_previous_interpretation(tmp_path, monkeypatch):
    """The reported hazard: a different provider or model silently reusing a fact set."""
    transport = ScriptedTransport(ok(expected_json()))
    _inject(monkeypatch, transport)
    store, client = _app(tmp_path, _hosted_config(tmp_path))
    path = _volunteer_submits(client, "/claims/upload")
    client.post(f"{path}/interpret", data={"csrf_token": _csrf(client)})
    assert len(transport.calls) == 1

    # Same claim, same evidence, different model — and a separate process would see the
    # same database, so this must be a fresh attempt rather than a redirect.
    other_store, other = _app(tmp_path, _hosted_config(tmp_path, model="gpt-4.1-nano"), store=store)
    other.post(
        "/login",
        data={
            "email": "volunteer@missionpay.local",
            "password": "volunteer-demo-passphrase",
            "csrf_token": _csrf(other, "/login"),
        },
        follow_redirects=False,
    )
    other.post(f"{path}/interpret", data={"csrf_token": _csrf(other)})
    assert len(transport.calls) == 2, "a different model must run inference, not reuse the last answer"

    history = other.get(path).text
    assert "gpt-4.1-nano" in history


def test_the_configuration_digest_separates_providers_with_the_same_model_name(tmp_path):
    from missionpay.domain import Activity, Claim, Money
    from datetime import date

    claim = Claim(
        claim_id="c1",
        volunteer_id="v1",
        activity=Activity("a1", "RIVER_CLEANUP", date(2026, 11, 3)),
        evidence=EVIDENCE,
        requested=Money("USD", 500),
        recipient="v@example.test",
        expense_category="PROTECTIVE_SUPPLIES",
    )
    hosted = web_module._generation_identity(_hosted_config(tmp_path))
    ollama = RuntimeConfig(ai_mode=AiMode.OLLAMA, ollama_model="gpt-4.1-mini")
    fixture = RuntimeConfig()
    digests = {
        web_module._interpretation_config_digest(hosted_config, claim, "gpt-4.1-mini")
        for hosted_config in (
            _hosted_config(tmp_path),
            RuntimeConfig(
                provider_mode=ProviderMode.MOCK,
                ai_mode=AiMode.HOSTED,
                hosted_model="gpt-4.1-mini",
                hosted_api_key=KEY,
                hosted_max_output_tokens=1024,  # a different ceiling, same model name
                hosted_max_total_tokens=8192,
                hosted_confirmed=True,
            ),
        )
    }
    assert len(digests) == 2, "output ceiling must separate two runs"
    assert hosted["provider"] == "openai"
    assert ollama.ai_mode is AiMode.OLLAMA
    assert fixture.ai_mode is AiMode.FIXTURE
