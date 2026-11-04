"""The prompt contract under v7: the model selects, it does not transcribe.

This file began as the v5 quote-context suite. Under the catalogue, the quote-level cases it
covered moved to `test_citation_resolution.py`, where the resolution code lives. What stays
here is what only the prompt can get wrong: what it asks for, what it must not reveal, and
whether the schema and the vocabulary agree with it.

Where the old cases went:

  repeated amount refused bare / accepted with its label -> test_citation_resolution.py
  context that is itself repeated                        -> test_citation_resolution.py
  non-contiguous quote assembled from two lines          -> not expressible: the model returns
                                                            no text; the catalogue is built from
                                                            whole contiguous lines
  overlapping / Unicode duplicates                       -> test_citation_resolution.py (tampered
                                                            catalogue, the layer that can produce them)
  unique quote that does not support the fact            -> test_citation_resolution.py, and the
                                                            verdict half stays in
                                                            test_qualification_comparator.py
"""

from __future__ import annotations

import pytest

from missionpay.ai import (
    FACT_NAME_DEFINITIONS,
    FACT_NAMES,
    INTERPRETATION_SCHEMA,
    PROMPT_VERSION,
    _SYSTEM_PROMPT,
    interpretation_prompt,
)
from missionpay.catalog import build_catalog

FIXTURE = "CANYON SUPPLY COOPERATIVE\nTOTAL PAID USD 5.00\n"


def test_the_contract_is_v7():
    assert PROMPT_VERSION == "missionpay-receipt-v7"


def test_the_prompt_asks_for_an_excerpt_identifier_and_forbids_transcription():
    text = _SYSTEM_PROMPT
    assert "excerpt_id" in text
    assert "Do not return a quote, start or end" in text
    assert "catalogue" in text.lower()


def test_the_prompt_says_an_identifier_outside_the_catalogue_is_refused():
    assert "outside the catalogue is refused" in _SYSTEM_PROMPT


def test_the_prompt_states_that_a_valid_excerpt_does_not_prove_support():
    assert "does not" in _SYSTEM_PROMPT
    assert "prove that the excerpt supports the fact" in _SYSTEM_PROMPT


def test_the_prompt_does_not_carry_answers_from_the_pinned_receipt():
    """A prompt that spells out expected values would make the benchmark prove nothing."""
    forbidden = ("CANYON", "canyon", "whistle", "lanyard", "4412", "5.00", "2.50", "2026-11-03")
    leaked = [token for token in forbidden if token in _SYSTEM_PROMPT]
    assert leaked == [], f"the prompt names values from the pinned receipt: {leaked}"


def test_the_three_total_names_are_defined_and_distinct():
    total = FACT_NAME_DEFINITIONS["total"]
    paid = FACT_NAME_DEFINITIONS["total_paid"]
    legacy = FACT_NAME_DEFINITIONS["total_usd"]
    assert "final amount actually paid" in paid
    assert "subtotal" not in total, "total keeps its printed meaning"
    assert "the receipt's total line" in total
    assert "prefer total_paid" in total
    assert "legacy" in legacy and "do not emit" in legacy
    assert "prefer total_paid" in _SYSTEM_PROMPT


def test_prompt_schema_and_definitions_share_one_vocabulary():
    item = INTERPRETATION_SCHEMA["properties"]["facts"]["items"]
    assert tuple(item["properties"]["name"]["enum"]) == FACT_NAMES
    assert set(FACT_NAMES) == set(FACT_NAME_DEFINITIONS)
    assert "excerpt_id" in item["required"]
    assert not {"start", "end", "quote"} & set(item["properties"])


def test_the_user_message_carries_the_catalogue_and_the_text():
    system, user = interpretation_prompt(FIXTURE)
    catalog = build_catalog(FIXTURE)
    assert "<receipt-text>" in user and "<excerpt-catalogue>" in user
    assert f"[{catalog.excerpts[0].excerpt_id}]" in user
    assert FIXTURE in user


def test_the_catalogue_in_the_prompt_never_offers_an_ambiguous_span():
    text = "Subtotal          5.00\nTOTAL PAID  5.00\n"
    _system, user = interpretation_prompt(text)
    catalogue_block = user.split("<excerpt-catalogue>")[1]
    # The bare amount occurs twice, so it must not appear as an entry line.
    entries = [line for line in catalogue_block.splitlines() if line.startswith("[")]
    assert entries
    assert all(line.split("] ", 1)[1] != "5.00" for line in entries)


def test_the_preflight_refuses_a_budget_below_the_prompt():
    """Growing the prompt must not have quietly disabled the pre-send check."""
    import dataclasses
    import tempfile
    from pathlib import Path

    from missionpay.hosted_qualify import QualificationRefused, build_plan, load_expectation
    from tests.test_hosted_corrections import _runtime, _seed_claim

    fixtures = Path(__file__).resolve().parent / "fixtures" / "hosted"
    with tempfile.TemporaryDirectory() as tmp:
        store, claim_id, _digest = _seed_claim(Path(tmp))
        config = dataclasses.replace(_runtime(Path(tmp)), hosted_max_total_tokens=1024)
        with pytest.raises(QualificationRefused) as excinfo:
            build_plan(
                store,
                config,
                authorization_id="MP-AI-PREFLIGHT",
                claim_id=claim_id,
                revision=1,
                expectation=load_expectation(fixtures),
            )
    assert excinfo.value.code == "HOSTED_REQUEST_OVER_BUDGET"
