"""One definition of "this text identifies exactly one place", shared by builder and resolver.

The catalogue decides which excerpts may be offered; the resolver decides whether a citation is
accepted. If those two disagree, a catalogue entry exists that the resolver will refuse — an
entry the model can choose and then be failed for choosing. MP-AI-004's reviewer found exactly
that shape of inconsistency elsewhere, so the rule lives here once.

Two ways a span can fail to identify one place:

* the same text occurs more than once, counting **overlapping** occurrences ("abab" sits at
  both 0 and 2 in "ababab", though ``str.count`` says 1);
* the same visible text occurs more than once once **NFC-normalised**, so a reader cannot tell
  "Café" (composed) from "Café" (decomposed). Normalisation is used only to detect this; it is
  never used to match.
"""

from __future__ import annotations

import unicodedata


def quote_occurrences(text: str, quote: str) -> list[int]:
    """Every start position where ``quote`` occurs in ``text``, overlaps included."""
    if not quote:
        return []
    positions: list[int] = []
    start = text.find(quote)
    while start != -1:
        positions.append(start)
        start = text.find(quote, start + 1)
    return positions


def unicode_twins(text: str, quote: str) -> int:
    """How many times ``quote`` occurs once both sides are NFC-normalised."""
    normalised_text = unicodedata.normalize("NFC", text)
    normalised_quote = unicodedata.normalize("NFC", quote)
    if not normalised_quote:
        return 0
    return len(quote_occurrences(normalised_text, normalised_quote))


def identifies_one_place(text: str, quote: str) -> bool:
    """The eligibility rule, in one place.

    A span is eligible only when it occurs exactly once verbatim AND exactly once as visible
    text. Both halves matter: the first is the exact-match guarantee, the second stops a
    visually duplicated quote from being cited as though it were unique.
    """
    if not quote:
        return False
    if len(quote_occurrences(text, quote)) != 1:
        return False
    return unicode_twins(text, quote) <= 1
