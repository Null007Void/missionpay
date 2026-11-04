"""A frozen benchmark fails on facts it did not expect.

Independent review of candidate-023: the comparator collected unexpected facts, reported them
in `unexpected_facts`, and still returned `passed: True`. An answer could therefore state the
expected `total_paid` correctly and add a contradictory `total_usd` — a second claim about the
same money, cited to the same receipt — and qualify.

These are the cases that would have ridden along inside a passing verdict.
"""

from __future__ import annotations

import json

from missionpay.catalog import build_catalog
from missionpay.hosted_qualify import compare
from tests.test_hosted_corrections import EXPECTED, RECEIPT, _resolved_interpretation


def _answer(extra=(), mutate=None):
    """A resolved interpretation — what compare() actually judges under v7."""
    interpretation = _resolved_interpretation()
    if mutate is not None:
        mutate(interpretation)
    interpretation["facts"].extend(extra)
    return interpretation


def _by_name(payload, name):
    return next(f for f in payload["facts"] if f["name"] == name)


TOTAL_PAID = _by_name(_resolved_interpretation(), "total_paid")


def _conflicting(name, value="USD 9.00"):
    """A second statement about the same money, cited at the total_paid span."""
    return {
        "name": name,
        "value": value,
        "epistemic_status": "observed",
        "start": TOTAL_PAID["start"],
        "end": TOTAL_PAID["end"],
        "quote": TOTAL_PAID["quote"],
    }


def test_the_happy_path_still_passes():
    """Guard first: the frozen expectation, answered exactly, must qualify."""
    verdict = compare(_resolved_interpretation(), EXPECTED, RECEIPT)
    assert verdict["passed"] is True, verdict["mismatches"]
    assert verdict["unexpected_facts"] == []


def test_a_contradictory_extra_total_fails_qualification():
    answer = _answer(extra=[_conflicting("total", "USD 7.50")])
    verdict = compare(answer, EXPECTED, RECEIPT)
    assert verdict["passed"] is False
    assert verdict["unexpected_facts"] == ["total"]
    assert any(m.startswith("total:") and "not in the frozen expectation" in m
               for m in verdict["mismatches"])


def test_a_contradictory_extra_total_usd_fails_qualification():
    answer = _answer(extra=[_conflicting("total_usd", "USD 9.00")])
    verdict = compare(answer, EXPECTED, RECEIPT)
    assert verdict["passed"] is False
    assert verdict["unexpected_facts"] == ["total_usd"]
    assert any("total_usd:" in m for m in verdict["mismatches"])


def test_an_extra_fact_that_agrees_is_still_refused():
    """Agreement is not the test. The benchmark is frozen; an unlisted fact is unverified."""
    answer = _answer(extra=[_conflicting("total_usd", "5.00")])
    verdict = compare(answer, EXPECTED, RECEIPT)
    assert verdict["passed"] is False
    assert verdict["unexpected_facts"] == ["total_usd"]


def test_both_extra_totals_are_named_in_one_verdict():
    answer = _answer(
        extra=[_conflicting("total", "USD 7.50"), _conflicting("total_usd", "USD 9.00")]
    )
    verdict = compare(answer, EXPECTED, RECEIPT)
    assert verdict["passed"] is False
    assert verdict["unexpected_facts"] == ["total", "total_usd"]
    named = [m for m in verdict["mismatches"] if "frozen expectation" in m]
    assert len(named) == 2


def test_a_duplicated_expected_fact_still_fails():
    answer = _answer(extra=[dict(TOTAL_PAID)])
    verdict = compare(answer, EXPECTED, RECEIPT)
    assert verdict["passed"] is False
    assert any("stated 2 times" in m for m in verdict["mismatches"])


def test_an_unexpected_fact_does_not_excuse_a_bad_citation():
    """The new check adds a failure; it does not replace the existing ones."""
    answer = _answer(
        extra=[_conflicting("total_usd")],
        mutate=lambda p: _by_name(p, "merchant").update({"start": 0, "end": 3, "quote": "Can"}),
    )
    verdict = compare(answer, EXPECTED, RECEIPT)
    assert verdict["passed"] is False
    assert any("merchant: quote is not receipt text" in m for m in verdict["mismatches"])
    assert verdict["unexpected_facts"] == ["total_usd"]


def test_the_verdict_reports_exactly_what_it_judged():
    verdict = compare(_resolved_interpretation(), EXPECTED, RECEIPT)
    assert json.dumps(verdict["expectation_sha256"])[:1] == '"'
    assert verdict["citation_rule"] == EXPECTED["citation_rule"]
