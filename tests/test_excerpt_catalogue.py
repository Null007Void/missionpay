"""The catalogue builder itself: uniqueness, determinism, anchoring, bounds.

Every property the resolver relies on is constructed here, so it is tested here rather than
assumed. The resolver's use of these excerpts is covered in test_citation_resolution.py.
"""

from __future__ import annotations

import inspect

import pytest

from missionpay import catalog as catalog_module
from missionpay.catalog import (
    MAX_EXCERPTS,
    MAX_EXCERPT_CHARS,
    MAX_RENDER_CHARS,
    build_catalog,
    text_digest,
)
from missionpay.ai import build_catalog as reexported

RECEIPT = (
    "CANYON SUPPLY COOPERATIVE\n"
    "1280 River Road, Moab UT 84532\n"
    "Receipt CS-4471                    2026-11-03\n"
    "\n"
    "2 x Safety whistle, lanyard     @ 2.50      5.00\n"
    "Subtotal                                    5.00\n"
    "Sales tax (included)                        0.00\n"
    "TOTAL PAID                              USD 5.00\n"
    "\n"
    "Paid with personal debit card ending 4412\n"
    "Collected for the river cleanup volunteer safety kit\n"
)


# -- the rule that makes the design work -----------------------------------


def test_every_offered_excerpt_occurs_exactly_once():
    for entry in build_catalog(RECEIPT).excerpts:
        assert RECEIPT.count(entry.text) == 1, entry.text


def test_every_offered_excerpt_matches_its_own_span():
    for entry in build_catalog(RECEIPT).excerpts:
        assert RECEIPT[entry.start : entry.end] == entry.text


def test_an_amount_that_repeats_is_never_offered_bare():
    texts = [e.text for e in build_catalog(RECEIPT).excerpts]
    assert "5.00" not in texts
    assert "TOTAL PAID                              USD 5.00" in texts


def test_a_receipt_with_nothing_unique_offers_nothing():
    assert len(build_catalog("5.00\n5.00\n")) == 0


def test_a_repeated_line_needs_a_unique_neighbour_to_be_offered():
    """ "5.00\\n5.00" is unique as a string while pointing at both occurrences equally."""
    assert len(build_catalog("5.00\n5.00\n")) == 0
    with_context = "STORE A\n5.00\n5.00\n"
    texts = [e.text for e in build_catalog(with_context).excerpts]
    assert "5.00" not in texts
    assert any(t.startswith("STORE A") and "5.00" in t for t in texts)


def test_a_window_that_only_repeats_itself_is_not_offered():
    catalog = build_catalog("X\nX\nX\n")
    assert all(entry.text != "X\nX" for entry in catalog.excerpts)


# -- determinism -----------------------------------------------------------


def test_the_same_text_always_produces_the_same_catalogue():
    first = build_catalog(RECEIPT)
    second = build_catalog(RECEIPT)
    assert [(e.excerpt_id, e.start, e.end, e.text) for e in first.excerpts] == [
        (e.excerpt_id, e.start, e.end, e.text) for e in second.excerpts
    ]


def test_different_text_produces_a_different_digest():
    assert build_catalog(RECEIPT).digest != build_catalog(RECEIPT + "x").digest


def test_the_digest_is_the_digest_of_the_text():
    assert build_catalog(RECEIPT).digest == text_digest(RECEIPT)


# -- identifiers -----------------------------------------------------------


def test_identifiers_start_at_one_and_are_contiguous():
    ids = [e.excerpt_id for e in build_catalog(RECEIPT).excerpts]
    assert ids == list(range(1, len(ids) + 1))


def test_identifiers_are_ordered_by_position():
    catalog = build_catalog(RECEIPT)
    starts = [e.start for e in catalog.excerpts]
    assert starts == sorted(starts)


def test_lookup_returns_the_entry_and_refuses_nonsense():
    catalog = build_catalog(RECEIPT)
    entry = catalog.excerpts[0]
    assert catalog.get(entry.excerpt_id) is entry
    for bad in (0, -1, 10_000, "1", 1.0, None, True):
        assert catalog.get(bad) is None


# -- bounds ----------------------------------------------------------------


def test_the_rendered_catalogue_stays_inside_its_budget():
    catalog = build_catalog(RECEIPT)
    assert sum(len(e.text) for e in catalog.excerpts) <= MAX_RENDER_CHARS


def test_no_excerpt_is_longer_than_the_character_bound():
    long_line = "x" * (MAX_EXCERPT_CHARS + 50)
    catalog = build_catalog(f"{long_line}\nSHORT LINE\n")
    assert all(len(e.text) <= MAX_EXCERPT_CHARS for e in catalog.excerpts)


def test_the_entry_count_is_bounded():
    text = "\n".join(f"LINE {i}" for i in range(MAX_EXCERPTS + 80))
    assert len(build_catalog(text)) <= MAX_EXCERPTS


def test_the_render_is_numbered_and_plain():
    """Entries may span lines, so the render is checked per entry, not per line."""
    catalog = build_catalog(RECEIPT)
    rendered = catalog.render()
    assert rendered.startswith("[1] ")
    for entry in catalog.excerpts:
        assert f"[{entry.excerpt_id}] {entry.text}" in rendered


# -- independence from any expectation -------------------------------------


def test_the_builder_takes_only_text():
    """It must be impossible to steer the catalogue toward a frozen answer."""
    parameters = list(inspect.signature(build_catalog).parameters)
    assert parameters == ["text"]


def test_the_module_does_not_read_an_expectation():
    """Docstrings may discuss expectations; no executable line may touch one."""
    import ast

    tree = ast.parse(inspect.getsource(catalog_module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    node.body = node.body[1:] or [ast.Pass()]
    body = ast.unparse(tree).lower()
    assert "expected_facts" not in body
    assert "expectation" not in body


def test_an_answer_shaped_string_does_not_change_the_catalogue():
    """Text is text. A line that looks like JSON has no special path through the builder."""
    text = '{"name": "total_paid", "value": "USD 5.00"}\nOTHER LINE\n'
    catalog = build_catalog(text)
    assert [e.text for e in catalog.excerpts] == [
        '{"name": "total_paid", "value": "USD 5.00"}',
        '{"name": "total_paid", "value": "USD 5.00"}\nOTHER LINE',
        "OTHER LINE",
    ], "identifiers follow position, so the window sits between the two lines it spans"
    assert all(text.count(e.text) == 1 for e in catalog.excerpts)


# -- module surface --------------------------------------------------------


def test_the_ai_module_reexports_the_builder_tests_use():
    assert reexported is build_catalog


@pytest.mark.parametrize("text", ["", "\n", "\n\n\n", "   \n  "])
def test_blank_input_yields_an_empty_catalogue(text):
    assert len(build_catalog(text)) == 0
