"""Regressions for the candidate-015 review: expectation preflight, ceilings, bounds, dry run.

Each test fails against the reviewed code:

* an empty expectation compared equal to everything and was accepted by the loader;
* ``epistemic_status`` was never judged;
* the output ceiling was not checked, and usage was lost when a response was refused;
* the elapsed-time check was described as a deadline it did not implement;
* bytes/3 was presented as if it bounded the request;
* a paid send could start before ``--expect`` was even read.
"""

from __future__ import annotations

import argparse
import json
import re
from hashlib import sha256
from pathlib import Path

import pytest

from missionpay import cli
from missionpay.ai import FACT_NAMES, interpretation_prompt
from missionpay.hosted import HostedJsonModel, HostedModelError
from missionpay.hosted_qualify import (
    QualificationRefused,
    build_plan,
    compare,
    load_expectation,
    qualify,
)
from missionpay.store import MissionPayStore
from tests.test_hosted_corrections import (
    DIGEST,
    EXPECTED,
    FIXTURES,
    KEY,
    MODEL_ID,
    RECEIPT,
    ScriptedTransport,
    _runtime,
    _seed_claim,
    answer,
    good_payload,
    model,
    _resolved_interpretation,
)

RECEIPT_SHA256 = "08a7e2e9cdbe77f7b78eec594207bc4acdb76fccfbe80cf2bff68967d452d312"


def _write(tmp_path, payload) -> Path:
    path = tmp_path / "expected.json"
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
    )
    return path


def _mutated(**changes):
    data = json.loads(json.dumps(EXPECTED))
    data.update(changes)
    return data


# -- 1. the expectation is fully loaded and validated before any send --------


def test_the_published_expectation_loads_and_matches_the_fixture():
    expectation = load_expectation(FIXTURES)
    assert expectation["fixture"]["sha256"] == RECEIPT_SHA256
    assert [f["name"] for f in expectation["expected_facts"]] == [
        "merchant",
        "purchase_date",
        "item",
        "quantity",
        "unit_price",
        "total_paid",
        "payment_method",
        "activity_link",
    ]
    for fact in expectation["expected_facts"]:
        assert fact["name"] in FACT_NAMES
        assert fact["epistemic_status"] in ("observed", "inference", "uncertain")


@pytest.mark.parametrize(
    "content,code",
    [
        ("", "EXPECTATION_EMPTY"),
        ("   \n", "EXPECTATION_EMPTY"),
        ("{", "EXPECTATION_INVALID"),
        ("[]", "EXPECTATION_INVALID"),
        ("{}", "EXPECTATION_INVALID"),
        (None, "EXPECTATION_MISSING"),
    ],
)
def test_a_missing_empty_or_malformed_expectation_is_refused(tmp_path, content, code):
    if content is None:
        target = tmp_path / "nothing-here"
    else:
        target = _write(tmp_path, content)
    with pytest.raises(QualificationRefused) as excinfo:
        load_expectation(target)
    assert excinfo.value.code == code


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda d: d.pop("fixture"), "EXPECTATION_INCOMPLETE"),
        (lambda d: d["fixture"].update({"sha256": "not-a-digest"}), "EXPECTATION_INCOMPLETE"),
        (lambda d: d.update({"expected_facts": []}), "EXPECTATION_EMPTY"),
        (lambda d: d.pop("expected_facts"), "EXPECTATION_EMPTY"),
        (lambda d: d["expected_facts"][0].pop("epistemic_status"), "EXPECTATION_INCOMPLETE"),
        (lambda d: d["expected_facts"][0].update({"value": ""}), "EXPECTATION_INCOMPLETE"),
        (lambda d: d["expected_facts"][0].update({"anchor": ""}), "EXPECTATION_INCOMPLETE"),
        (lambda d: d["expected_facts"][0].update({"name": "not_a_fact"}), "EXPECTATION_UNKNOWN_FACT"),
        (lambda d: d["expected_facts"].append(dict(d["expected_facts"][0])), "EXPECTATION_INCOMPLETE"),
        (lambda d: d["expected_facts"][0].update({"canonical_start": "0"}), "EXPECTATION_INCOMPLETE"),
    ],
)
def test_an_incomplete_expectation_is_refused(tmp_path, mutate, code):
    data = json.loads(json.dumps(EXPECTED))
    mutate(data)
    with pytest.raises(QualificationRefused) as excinfo:
        load_expectation(_write(tmp_path, data))
    assert excinfo.value.code == code


def test_an_empty_expectation_can_never_pass_a_verdict():
    verdict = compare(_resolved_interpretation(), {}, RECEIPT)
    assert verdict["passed"] is False
    assert any("no expected facts" in item for item in verdict["mismatches"])


def test_an_expectation_for_another_receipt_is_refused_before_sending(tmp_path):
    store, claim_id, _digest = _seed_claim(tmp_path)
    other = _mutated()
    other["fixture"]["sha256"] = "b" * 64
    transport = ScriptedTransport(answer(json.dumps(good_payload())))
    with pytest.raises(QualificationRefused) as excinfo:
        qualify(
            store,
            _runtime(tmp_path),
            authorization_id="auth-1",
            claim_id=claim_id,
            expectation=other,
            model=model(transport),
        )
    assert excinfo.value.code == "EXPECTATION_RECEIPT_MISMATCH"
    assert transport.calls == []
    assert store.hosted_attempts() == []


def test_an_anchor_that_is_not_unique_in_the_receipt_is_refused(tmp_path):
    store, claim_id, _digest = _seed_claim(tmp_path)
    ambiguous = _mutated()
    # "5.00" appears twice in the receipt, so citing it cannot identify the evidence.
    for fact in ambiguous["expected_facts"]:
        if fact["name"] == "total_paid":
            fact["anchor"] = "5.00"
    transport = ScriptedTransport(answer(json.dumps(good_payload())))
    with pytest.raises(QualificationRefused) as excinfo:
        qualify(
            store,
            _runtime(tmp_path),
            authorization_id="auth-1",
            claim_id=claim_id,
            expectation=ambiguous,
            model=model(transport),
        )
    assert excinfo.value.code == "EXPECTATION_ANCHOR_NOT_UNIQUE"
    assert transport.calls == []


def test_the_plan_pins_the_expectation_digest_and_the_time_limits(tmp_path):
    store, claim_id, _digest = _seed_claim(tmp_path)
    plan, _text, _evidence = build_plan(
        store,
        _runtime(tmp_path),
        authorization_id="auth-1",
        claim_id=claim_id,
        revision=None,
        expectation=EXPECTED,
        model=model(),
    )
    assert plan.expectation_sha256 == sha256(
        json.dumps(EXPECTED, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert plan.timeout_seconds == 45
    assert plan.late_response_seconds == 45
    as_dict = plan.as_dict()
    assert as_dict["expectation_sha256"] == plan.expectation_sha256
    assert as_dict["late_response_seconds"] == 45

    # A different expectation must produce a different request identity.
    widened = _mutated()
    widened["comparison_rule"] = widened["comparison_rule"] + " (changed)"
    other, _t, _e = build_plan(
        store,
        _runtime(tmp_path),
        authorization_id="auth-1",
        claim_id=claim_id,
        revision=None,
        expectation=widened,
        model=model(),
    )
    assert other.config_sha256 != plan.config_sha256
    assert other.expectation_sha256 != plan.expectation_sha256


def test_the_time_limits_are_pinned_in_the_request_identity(tmp_path):
    store, claim_id, _digest = _seed_claim(tmp_path)
    base, _t, _e = build_plan(
        store, _runtime(tmp_path), authorization_id="a", claim_id=claim_id,
        revision=None, expectation=EXPECTED, model=model(),
    )
    slower, _t, _e = build_plan(
        store, _runtime(tmp_path), authorization_id="a", claim_id=claim_id,
        revision=None, expectation=EXPECTED, model=model(late_response_seconds=120),
    )
    assert slower.config_sha256 != base.config_sha256, "a longer late-response window is a different request"
    assert slower.late_response_seconds == 120


# -- 2. epistemic status is judged -------------------------------------------


def test_an_inference_reported_as_observed_fails_the_verdict():
    interpretation = _resolved_interpretation()
    for fact in interpretation["facts"]:
        fact["epistemic_status"] = "uncertain"
    verdict = compare(interpretation, EXPECTED, RECEIPT)
    assert verdict["passed"] is False
    assert all("status" in item for item in verdict["mismatches"][:1])
    assert len(verdict["mismatches"]) == len(EXPECTED["expected_facts"])


def test_a_correct_status_still_passes():
    assert compare(_resolved_interpretation(), EXPECTED, RECEIPT)["passed"] is True


# -- 3. provider completion is not qualification ----------------------------


def test_a_completed_request_that_fails_the_verdict_is_not_reported_as_qualified(tmp_path):
    store, claim_id, _digest = _seed_claim(tmp_path)
    wrong = good_payload()
    for fact in wrong["facts"]:
        if fact["name"] == "total_paid":
            fact["value"] = "USD 50.00"
    transport = ScriptedTransport(answer(json.dumps(wrong)))
    outcome = qualify(
        store,
        _runtime(tmp_path),
        authorization_id="auth-1",
        claim_id=claim_id,
        expectation=EXPECTED,
        model=model(transport),
    )
    assert outcome["status"] == "VERDICT_FAILED"
    assert outcome["provider_completed"] is True
    assert outcome["sent"] is True
    assert outcome["verdict"]["passed"] is False
    # The provider did answer, so the stored attempt is still SUCCEEDED: two different facts.
    assert store.hosted_attempt("auth-1")["state"] == "SUCCEEDED"


def test_a_completed_and_matching_request_is_qualified(tmp_path):
    store, claim_id, _digest = _seed_claim(tmp_path)
    outcome = qualify(
        store,
        _runtime(tmp_path),
        authorization_id="auth-1",
        claim_id=claim_id,
        expectation=EXPECTED,
        model=model(ScriptedTransport(answer(json.dumps(good_payload())))),
    )
    assert outcome["status"] == "QUALIFIED"
    assert outcome["provider_completed"] is True
    assert outcome["verdict"]["passed"] is True


def _cli_args(**overrides):
    base = dict(
        dry_run=False,
        confirm=True,
        authorization="auth-cli",
        claim="claim-live",
        revision=1,
        expect=str(FIXTURES),
        json=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_the_cli_exits_three_when_the_verdict_fails_and_zero_when_it_passes(
    tmp_path, monkeypatch, capsys
):
    store, _claim_id, _digest = _seed_claim(tmp_path)
    config = _runtime(tmp_path)

    failing = ScriptedTransport(answer(json.dumps(good_payload())))
    monkeypatch.setattr(cli, "_config", lambda: config)
    monkeypatch.setattr(
        "missionpay.hosted_qualify.build_model", lambda _config: model(failing)
    )
    exit_code = cli._hosted_qualify(_cli_args(authorization="auth-pass"))
    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "QUALIFICATION PASS" in printed
    assert "PROVIDER_STATUS QUALIFIED" in printed


def test_the_cli_reports_a_failed_verdict_as_qualification_failure(tmp_path, monkeypatch, capsys):
    store, _claim_id, _digest = _seed_claim(tmp_path)
    config = _runtime(tmp_path)
    wrong = good_payload()
    for fact in wrong["facts"]:
        if fact["name"] == "merchant":
            fact["value"] = "CANYON SUPPLY"
    transport = ScriptedTransport(answer(json.dumps(wrong)))
    monkeypatch.setattr(cli, "_config", lambda: config)
    monkeypatch.setattr("missionpay.hosted_qualify.build_model", lambda _config: model(transport))
    exit_code = cli._hosted_qualify(_cli_args())
    assert exit_code == 3
    captured = capsys.readouterr()
    assert "QUALIFICATION FAIL" in captured.err
    assert "QUALIFICATION PASS" not in captured.out


def test_a_missing_expectation_costs_nothing(tmp_path, monkeypatch, capsys):
    store, _claim_id, _digest = _seed_claim(tmp_path)
    config = _runtime(tmp_path)
    transport = ScriptedTransport(answer(json.dumps(good_payload())))
    monkeypatch.setattr(cli, "_config", lambda: config)
    monkeypatch.setattr("missionpay.hosted_qualify.build_model", lambda _config: model(transport))
    exit_code = cli._hosted_qualify(_cli_args(expect=str(tmp_path / "no-such-file")))
    assert exit_code == 2
    assert "EXPECTATION_MISSING" in capsys.readouterr().err
    assert transport.calls == []
    assert MissionPayStore(store.path).hosted_attempts() == []


# -- 4. the output ceiling, and usage that survives a refusal ---------------


def test_more_output_tokens_than_the_ceiling_is_refused_but_the_usage_is_kept():
    usage = {"prompt_tokens": 980, "completion_tokens": 600, "total_tokens": 1580}
    instance = model(
        ScriptedTransport(answer(json.dumps(good_payload()), usage=usage)),
        max_output_tokens=512,
    )
    with pytest.raises(HostedModelError) as excinfo:
        instance.complete_json("s", "u")
    assert excinfo.value.code == "HOSTED_OUTPUT_CEILING_EXCEEDED"
    assert instance.last_usage == usage, "a refused answer still cost money"
    assert len(instance._transport.calls) == 1


def test_usage_over_the_total_budget_is_refused_and_kept():
    usage = {"prompt_tokens": 8000, "completion_tokens": 512, "total_tokens": 8512}
    instance = model(ScriptedTransport(answer(json.dumps(good_payload()), usage=usage)))
    with pytest.raises(HostedModelError) as excinfo:
        instance.complete_json("s", "u")
    assert excinfo.value.code == "HOSTED_USAGE_OVER_BUDGET"
    assert instance.last_usage == usage


def test_the_ceiling_is_checked_before_the_budget():
    usage = {"prompt_tokens": 4000, "completion_tokens": 600, "total_tokens": 4600}
    instance = model(ScriptedTransport(answer(json.dumps(good_payload()), usage=usage)))
    with pytest.raises(HostedModelError) as excinfo:
        instance.complete_json("s", "u")
    assert excinfo.value.code == "HOSTED_OUTPUT_CEILING_EXCEEDED"
    assert instance.last_usage == usage


# -- 5. bounds are labelled honestly -----------------------------------------


def test_the_pre_send_bound_is_a_byte_bound_not_an_estimate():
    estimate = model().preview(*interpretation_prompt(RECEIPT))
    assert estimate["upper_bound_input_tokens"] == estimate["payload_bytes"]
    assert estimate["upper_bound_total_tokens"] == (
        estimate["payload_bytes"] + estimate["max_output_tokens"]
    )
    assert estimate["estimated_input_tokens"] < estimate["upper_bound_input_tokens"]
    # The basis must describe an assumption, not assert a property of the tokenizer.
    assert "assumption, not a proven bound" in estimate["bound_basis"]
    assert "not a bound" in estimate["estimate_basis"]


def test_a_budget_below_the_byte_bound_is_refused_before_any_call():
    transport = ScriptedTransport(answer(json.dumps(good_payload())))
    estimate = model().preview(*interpretation_prompt(RECEIPT))
    budget = estimate["upper_bound_total_tokens"] - 1
    instance = model(transport, max_total_tokens=budget)
    with pytest.raises(HostedModelError) as excinfo:
        instance.complete_json(*interpretation_prompt(RECEIPT))
    assert excinfo.value.code == "HOSTED_REQUEST_OVER_BUDGET"
    assert transport.calls == []


def test_the_late_response_check_is_named_for_what_it_is(monkeypatch):
    class Clock:
        def __init__(self):
            self.reads = 0

        def __call__(self):
            self.reads += 1
            return 100.0 if self.reads == 1 else 146.0

    monkeypatch.setattr("missionpay.hosted.time.monotonic", Clock())
    instance = model(ScriptedTransport(answer(json.dumps(good_payload()))), late_response_seconds=45)
    with pytest.raises(HostedModelError) as excinfo:
        instance.complete_json("s", "u")
    assert excinfo.value.code == "HOSTED_LATE_RESPONSE"
    described = instance.describe()
    assert "late_response_seconds" in described
    assert "deadline_seconds" not in described


# -- 6. the dry run uses the real $5 claim -----------------------------------


def test_the_dry_run_pins_the_real_five_dollar_claim(tmp_path):
    store, claim_id, digest = _seed_claim(tmp_path)
    assert digest == RECEIPT_SHA256, "the seeded evidence is the synthetic $5 receipt"
    plan, text, evidence = build_plan(
        store,
        _runtime(tmp_path),
        authorization_id="auth-dry",
        claim_id=claim_id,
        revision=1,
        expectation=load_expectation(FIXTURES),
        model=model(),
    )
    assert evidence.sha256 == RECEIPT_SHA256
    assert text == RECEIPT
    assert plan.evidence_sha256 == RECEIPT_SHA256
    assert plan.revision == 1
    assert plan.model_id == f"openai:{MODEL_ID}"
    assert plan.estimate["payload_bytes"] > 2000
    assert plan.max_output_tokens == 512
    assert plan.max_total_tokens == 8192
    assert plan.estimate["upper_bound_total_tokens"] <= plan.max_total_tokens
    assert store.hosted_attempts() == []
    # The plan is JSON-serialisable evidence with no secret in it.
    rendered = json.dumps(plan.as_dict(), sort_keys=True)
    assert KEY not in rendered
    assert "api_key" not in rendered


# -- 7. concurrency: two runs, one send -------------------------------------


def test_two_concurrent_runs_under_one_authorisation_send_once(tmp_path):
    """The journal is the lock, not a flag: the database decides, under contention."""
    import concurrent.futures
    import threading

    store, claim_id, _digest = _seed_claim(tmp_path)
    config = _runtime(tmp_path)
    lock = threading.Lock()
    states: list[str] = []

    class Recording:
        def __init__(self):
            self.calls = 0

        def request(self, method, url, headers, body, *, max_bytes, timeout):
            with lock:
                self.calls += 1
                row = store.hosted_attempt("auth-race")
                states.append("missing" if row is None else str(row["state"]))
            return answer(json.dumps(good_payload()))

    transport = Recording()
    barrier = threading.Barrier(2)

    def run():
        barrier.wait()
        return qualify(
            store,
            config,
            authorization_id="auth-race",
            claim_id=claim_id,
            expectation=EXPECTED,
            model=model(transport),
        )["status"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        results: list[str] = sorted(f.result(timeout=30) for f in futures)

    assert results == ["ALREADY_ATTEMPTED", "QUALIFIED"]
    assert transport.calls == 1, "two runs, one request"
    assert states == ["RECORDED"], "the row existed before the one send"
