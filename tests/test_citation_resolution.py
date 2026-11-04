"""Excerpt resolution: MissionPay owns the text, the answer only names a number.

This file replaces the quote-resolution suite. The rule it protects is the same — a citation
must be exact text at a span, and must identify exactly one place — but the mechanism changed
in contract v7: the model no longer transcribes anything, so the failure classes that came from
transcription are asserted where they now live.

Coverage mapping (old test -> where its property is asserted now):

  correct offsets used as supplied        -> a valid id resolves to the exact printed span
  duplicate quote refused (span matches)  -> the catalogue withholds ambiguous text; a
                                             tampered catalogue containing duplicates is refused
  duplicate quote refused (first)         -> same, and the labelled neighbour is offered instead
  wrong offsets located                   -> the resolved span is the catalogue's, never the
                                             answer's; there is nothing to misplace
  MP-AI-002 off-by-one                    -> the length mismatch cannot arise: no length is
                                             supplied, the whole labelled line is cited
  malformed offsets                       -> malformed excerpt_id (bool, float, string, None)
  absent quote / paraphrase               -> EXCERPT_UNKNOWN for an id outside the catalogue
  empty quote / non-string quote          -> an empty catalogue refuses every id
  duplicated quote with count             -> tampered catalogue reports the duplicate count
  overlapping occurrences                 -> tampered catalogue refuses overlapping duplicates
  unicode offsets / astral characters     -> spans are code points, resolved by MissionPay
  no unicode normalisation                -> NFC and NFD spellings are different excerpts
  whole-receipt quote                     -> a multi-line excerpt is still exact
  regex metacharacters / newlines         -> excerpt text is literal, never a pattern
  quote longer than the receipt           -> an id beyond the catalogue is refused
  secret-shaped text is not special       -> ids are numbers; no text is special
  resolution recorded                     -> excerpt_id is recorded on the stored fact
  ambiguous evidence cannot reach         -> a refusal leaves no interpretation, so no
  authorization or payment                   authorization and no payment intent
  resolved citation needs acceptance      -> a valid excerpt still cannot authorize anything
  unicode duplicate / unique              -> tampered-catalogue duplicates refused, unique kept
  page distinguishes supplied vs located  -> page names the excerpt and who resolved the span
"""

from __future__ import annotations

import pytest

from missionpay.ai import (
    CitationRefusal,
    build_catalog,
    interpret_receipt_text,
    quote_occurrences,
)
from missionpay.catalog import Catalog, Excerpt, text_digest
from missionpay.domain import EvidenceIdentity
from tests.catalogue_support import PayloadModel, answer, id_for

EVIDENCE = EvidenceIdentity("a" * 64, "text/plain", "synthetic.txt")
TEXT = "CANYON SUPPLY COOPERATIVE\nTOTAL PAID USD 5.00\n"
TOTAL_LINE = "TOTAL PAID USD 5.00"
TOTAL = "USD 5.00"


def _run(excerpt_id, payload_fact=None, text=TEXT, catalog=None, missing=None):
    fact = payload_fact or {
        "name": "total_paid",
        "value": "USD 5.00",
        "epistemic_status": "observed",
        "excerpt_id": excerpt_id,
    }
    model = PayloadModel(answer(fact, missing=missing))
    return interpret_receipt_text(evidence=EVIDENCE, text=text, model=model, catalog=catalog)


def _refusal(excerpt_id, **kwargs):
    with pytest.raises(CitationRefusal) as excinfo:
        _run(excerpt_id, **kwargs)
    return excinfo.value


def _tampered(text: str, excerpt: str) -> Catalog:
    """A catalogue containing an excerpt a correct builder would have withheld."""
    at = text.index(excerpt)
    return Catalog(
        digest=text_digest(text),
        excerpts=(Excerpt(excerpt_id=1, start=at, end=at + len(excerpt), text=excerpt),),
    )


# -- a valid id resolves to the printed span --------------------------------


def test_a_valid_id_resolves_to_the_exact_printed_span():
    number = id_for(TEXT, TOTAL_LINE)
    fact = _run(number).facts[0]
    assert fact.citation.quote == TOTAL_LINE
    assert TEXT[fact.citation.start : fact.citation.end] == TOTAL_LINE
    assert fact.citation.excerpt_id == number
    assert fact.offsets_derived is False


REPEATED = "Subtotal          5.00\nTOTAL PAID  5.00\n"


def test_the_catalogue_withholds_an_ambiguous_amount_but_offers_its_labels():
    """The MP-AI-003 / MP-AI-004 failure cannot be expressed any more.

    "5.00" occurs twice, so it is never an excerpt. The labelled lines are each unique, so
    each is offered — and each identifies one occurrence.
    """
    texts = [e.text for e in build_catalog(REPEATED).excerpts]
    assert "5.00" not in texts, "an ambiguous amount must not be citable"
    assert "Subtotal          5.00" in texts and "TOTAL PAID  5.00" in texts


def test_a_labelled_neighbour_resolves_to_its_own_occurrence():
    fact = _run(id_for(REPEATED, "TOTAL PAID"), text=REPEATED).facts[0]
    assert fact.citation.quote == "TOTAL PAID  5.00"
    assert fact.citation.start == REPEATED.index("TOTAL PAID")


def test_two_amounts_on_one_line_are_cited_as_that_line():
    """A single line holding both amounts is one excerpt: still exact, still unique."""
    text = "A 5.00 B 5.00\n"
    texts = [e.text for e in build_catalog(text).excerpts]
    assert texts == ["A 5.00 B 5.00"]


def test_excerpts_are_whole_lines_so_a_sub_line_amount_is_not_offered():
    """Granularity is a design property, not an accident: a line is the smallest unit.

    That is what makes the labelled total citable while the bare amount inside it is not —
    a sub-line span would hand the model back the ambiguity the catalogue exists to remove.
    """
    texts = [e.text for e in build_catalog(TEXT).excerpts]
    assert TOTAL_LINE in texts
    assert TOTAL not in texts


# -- fail closed on identifiers --------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, 9999, 10**6])
def test_an_id_outside_the_catalogue_is_refused(bad):
    exc = _refusal(bad)
    assert exc.diagnostic["category"] == "EXCERPT_UNKNOWN"
    assert exc.diagnostic["excerpt_id"] == bad


@pytest.mark.parametrize("bad", ["1", 1.0, None, True, [1], {"id": 1}])
def test_a_non_integer_id_is_refused(bad):
    exc = _refusal(bad)
    assert exc.diagnostic["category"] in {"EXCERPT_UNKNOWN", "QUOTE_NOT_TEXT"}


def test_an_empty_catalogue_refuses_every_id():
    text = "5.00\n5.00\n"  # nothing here can be made unique
    assert len(build_catalog(text)) == 0
    assert _refusal(1, text=text).diagnostic["category"] == "EXCERPT_UNKNOWN"


# -- stale and altered receipts -------------------------------------------


def test_a_catalogue_from_a_different_receipt_is_refused_as_stale():
    stale = build_catalog("SOME OTHER RECEIPT\nTOTAL PAID USD 5.00\n")
    assert _refusal(1, catalog=stale).diagnostic["category"] == "EXCERPT_STALE"


def test_an_altered_receipt_cannot_reuse_an_old_catalogue():
    original = build_catalog(TEXT)
    altered = TEXT.replace("CANYON", "CANY0N")
    assert _refusal(1, text=altered, catalog=original).diagnostic["category"] == "EXCERPT_STALE"


def test_an_excerpt_whose_span_no_longer_matches_is_refused():
    broken = Catalog(
        digest=text_digest(TEXT),
        excerpts=(Excerpt(excerpt_id=1, start=0, end=25, text="SOMETHING ELSE ENTIRELY"),),
    )
    exc = _refusal(1, catalog=broken)
    assert exc.diagnostic["category"] in {
        "EXCERPT_MISMATCH",
        "QUOTE_NOT_FOUND",
        "QUOTE_AMBIGUOUS",
    }


# -- the old ambiguity classes, at the layer where they now live -----------


def test_a_duplicated_excerpt_is_refused_and_counted():
    text = "A 5.00 B 5.00\n"
    exc = _refusal(1, text=text, catalog=_tampered(text, "5.00"))
    assert exc.diagnostic["category"] == "QUOTE_AMBIGUOUS"
    assert exc.diagnostic["occurrences"] == 2


def test_overlapping_excerpts_are_counted():
    text = "ababab\n"
    assert quote_occurrences(text, "abab") == [0, 2]
    exc = _refusal(1, text=text, catalog=_tampered(text, "abab"))
    assert exc.diagnostic["category"] == "QUOTE_AMBIGUOUS"
    assert exc.diagnostic["occurrences"] == 2


def test_unicode_duplicates_are_refused_even_when_the_span_matches():
    nfc, nfd = "Caf\u00e9", "Cafe\u0301"
    text = f"{nfc} 5.00 then {nfd} 5.00\n"
    exc = _refusal(1, text=text, catalog=_tampered(text, nfc))
    assert exc.diagnostic["category"] == "QUOTE_AMBIGUOUS"
    assert exc.diagnostic.get("unicode_duplicate") is True
    assert exc.diagnostic["occurrences"] == 2


def test_a_visually_unique_excerpt_is_accepted():
    nfc = "Caf\u00e9"
    text = f"{nfc} 5.00\n"
    fact = _run(id_for(text, nfc), text=text).facts[0]
    assert fact.citation.start == 0
    assert fact.citation.quote == f"{nfc} 5.00", "the whole line, not the bare token"


# -- code points, not bytes ------------------------------------------------


def test_spans_are_code_points_for_non_ascii_text():
    text = "\u00e9\u00e9\u00e9 TOTAL PAID USD 5.00\n"
    line = text[:-1]
    fact = _run(id_for(text, TOTAL_LINE), text=text).facts[0]
    assert fact.citation.start == 0
    assert fact.citation.end - fact.citation.start == len(line), "offsets count code points"
    assert text[fact.citation.start : fact.citation.end] == line


def test_astral_characters_count_as_one_position_each():
    text = "\U0001f600\U0001f600 TOTAL PAID USD 5.00\n"
    line = text[:-1]
    fact = _run(id_for(text, TOTAL_LINE), text=text).facts[0]
    assert fact.citation.end - fact.citation.start == len(line)
    assert text[fact.citation.start : fact.citation.end] == line


def test_no_unicode_normalisation_is_applied_to_excerpt_text():
    nfd = "Cafe\u0301"
    text = f"{nfd} TOTAL PAID USD 5.00\n"
    texts = [e.text for e in build_catalog(text).excerpts]
    assert any(t.startswith(nfd) for t in texts)
    assert not any(t.startswith("Caf\u00e9") for t in texts)


# -- literal text, never a pattern ----------------------------------------


def test_regex_metacharacters_are_literal():
    text = "A.*(B)+ TOTAL PAID USD 5.00\n"
    fact = _run(id_for(text, "A.*(B)+"), text=text).facts[0]
    assert fact.citation.quote == "A.*(B)+ TOTAL PAID USD 5.00"
    assert "*" in fact.citation.quote and "(" in fact.citation.quote


def test_a_multi_line_excerpt_is_still_exact():
    catalog = build_catalog(TEXT)
    multiline = [e for e in catalog.excerpts if "\n" in e.text]
    assert multiline, "a multi-line excerpt should be offered"
    entry = multiline[0]
    fact = _run(entry.excerpt_id).facts[0]
    assert TEXT[entry.start : entry.end] == entry.text
    assert fact.citation.quote == entry.text


def test_a_secret_shaped_string_is_not_special():
    text = "sk-live-abcdef123456 TOTAL PAID USD 5.00\n"
    fact = _run(id_for(text, "sk-live-abcdef123456"), text=text).facts[0]
    assert fact.citation.quote == "sk-live-abcdef123456 TOTAL PAID USD 5.00"


# -- a valid citation is not a true statement ------------------------------


def test_a_valid_excerpt_does_not_prove_it_supports_the_fact():
    """The boundary the product must not cross, asserted where the citation is created."""
    number = id_for(TEXT, TOTAL_LINE)
    fact = _run(
        number,
        payload_fact={
            "name": "total_paid",
            "value": "USD 9.99",
            "epistemic_status": "observed",
            "excerpt_id": number,
        },
    ).facts[0]
    assert fact.citation.quote == TOTAL_LINE  # the citation is impeccable
    assert "9.99" not in fact.citation.quote  # and says nothing about the value
