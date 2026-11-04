"""Eligibility is one rule, applied in both places.

Independent review of candidate-025: the builder counted with ``str.count`` (exact,
non-overlapping) while the resolver counted overlaps and Unicode twins. A span could therefore
be offered that the resolver refuses — an entry the model may choose and then be failed for.
These tests pin the two implementations to one predicate, and check the properties that follow.
"""

from __future__ import annotations

import json

import pytest

from missionpay.ai import CitationRefusal, build_catalog, interpret_receipt_text
from missionpay.domain import EvidenceIdentity
from missionpay.textmatch import identifies_one_place, quote_occurrences, unicode_twins
from tests.catalogue_support import PayloadModel, answer

EVIDENCE = EvidenceIdentity("a" * 64, "text/plain", "synthetic.txt")

CORPUS = [
    "CANYON SUPPLY COOPERATIVE\nTOTAL PAID USD 5.00\n",
    "Subtotal          5.00\nTOTAL PAID  5.00\n",
    "ababab\n",
    "X abab\nY abab\n",
    "Caf\u00e9 5.00\nCafe\u0301 5.00\n",
    "\U0001f600 TOTAL PAID USD 5.00\n\U0001f600 OTHER\n",
    "5.00\n5.00\n",
    "   \n\n",
    "A 5.00 B 5.00\n",
]


# -- the shared predicate ---------------------------------------------------


def test_overlapping_occurrences_are_counted_by_the_predicate():
    assert quote_occurrences("ababab", "abab") == [0, 2]
    assert not identifies_one_place("ababab", "abab")


def test_unicode_twins_are_counted_by_the_predicate():
    text = "Caf\u00e9 5.00 and Cafe\u0301 5.00"
    assert unicode_twins(text, "Caf\u00e9") == 2
    assert not identifies_one_place(text, "Caf\u00e9")


def test_an_empty_quote_identifies_nothing():
    assert quote_occurrences("abc", "") == []
    assert not identifies_one_place("abc", "")


def test_a_plain_unique_span_passes():
    assert identifies_one_place("A 5.00 B 3.00", "A 5.00")


# -- the builder applies it -------------------------------------------------


def test_an_overlapping_span_is_never_offered():
    """`str.count("abab")` in "ababab" is 1; it occurs twice. It must not be citable."""
    texts = [e.text for e in build_catalog("ababab\n").excerpts]
    assert "abab" not in texts


def test_unicode_twin_lines_are_both_withheld():
    """Two spellings a reader cannot tell apart make neither of them evidence."""
    text = "Caf\u00e9 5.00\nCafe\u0301 5.00\n"
    texts = [e.text for e in build_catalog(text).excerpts]
    assert "Caf\u00e9 5.00" not in texts
    assert "Cafe\u0301 5.00" not in texts
    assert texts == [], "the whole receipt contains the twin too, so nothing is citable"


def test_a_unique_spelling_is_still_offered():
    text = "Caf\u00e9 5.00\nOTHER LINE\n"
    texts = [e.text for e in build_catalog(text).excerpts]
    assert "Caf\u00e9 5.00" in texts


def test_a_window_is_anchored_by_a_line_that_actually_identifies_one_place():
    text = "STORE A\n5.00\n5.00\n"
    catalog = build_catalog(text)
    texts = [e.text for e in catalog.excerpts]
    assert "5.00" not in texts
    assert any(t.startswith("STORE A") and "5.00" in t for t in texts)


def test_a_window_never_becomes_unique_only_by_overlap():
    """A window whose text overlaps itself is still not two places' worth of evidence."""
    text = "abab\nabab\nX\n"
    for entry in build_catalog(text).excerpts:
        assert quote_occurrences(text, entry.text) == [text.index(entry.text)]


# -- every naturally offered identifier resolves ----------------------------


@pytest.mark.parametrize("text", CORPUS)
def test_every_offered_identifier_resolves(text):
    """The property the reviewer asked for: nothing is offered that the resolver refuses."""
    catalog = build_catalog(text)
    for entry in catalog.excerpts:
        model = PayloadModel(
            answer(
                {
                    "name": "total_paid",
                    "value": "5.00",
                    "epistemic_status": "observed",
                    "excerpt_id": entry.excerpt_id,
                }
            )
        )
        try:
            result = interpret_receipt_text(evidence=EVIDENCE, text=text, model=model)
        except CitationRefusal as exc:  # pragma: no cover - the failure this test exists for
            pytest.fail(
                f"offered id {entry.excerpt_id} ({entry.text!r}) was refused: "
                f"{exc.diagnostic.get('category')}"
            )
        fact = result.facts[0]
        assert fact.citation.excerpt_id == entry.excerpt_id
        assert text[fact.citation.start : fact.citation.end] == entry.text
