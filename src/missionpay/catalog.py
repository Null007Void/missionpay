"""A numbered catalogue of exact receipt excerpts.

The model no longer copies text. It picks an identifier from a catalogue this module builds
deterministically from the receipt, and the identifier is resolved back to a span of the
original text before the existing exact-match and uniqueness rules run.

Why: MP-AI-003 and MP-AI-004 both failed the same way — the model offered a four-character
money string that occurs three times. Asking it more firmly to "quote enough context" did not
change what it produced. Removing the transcription step removes the class of failure, because
the excerpt text is produced here, not by the model, and only excerpts that occur exactly once
enter the catalogue at all.

Nothing here reads an expectation. The catalogue is a function of the receipt text alone, so it
cannot be steered toward a frozen answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .textmatch import identifies_one_place, quote_occurrences

#: Bounded so the prompt cannot grow without limit on a long receipt.
MAX_EXCERPTS = 160

#: A window is this many consecutive lines. One line is the common case; two to four let a
#: repeated line become unique by carrying its neighbours.
WINDOW_SIZE = (1, 2, 3, 4)

#: Excerpts longer than this are line blocks nobody needs, and they crowd out finer ones.
MAX_EXCERPT_CHARS = 220

#: Total characters of excerpt text rendered into the prompt. Chosen so the whole request stays
#: inside the pre-send comparison; entries are picked shortest-first, which favours single
#: lines and maximises how many spans stay citable, and never depends on what is expected.
MAX_RENDER_CHARS = 1400


@dataclass(frozen=True, slots=True)
class Excerpt:
    excerpt_id: int
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class Catalog:
    """The excerpts, plus the digest of the text they were cut from."""

    digest: str
    excerpts: tuple[Excerpt, ...]

    def get(self, excerpt_id: object) -> Excerpt | None:
        if isinstance(excerpt_id, bool) or not isinstance(excerpt_id, int):
            return None
        for excerpt in self.excerpts:
            if excerpt.excerpt_id == excerpt_id:
                return excerpt
        return None

    def render(self) -> str:
        """The catalogue as the model sees it: identifier and text, nothing else."""
        return "\n".join(f"[{e.excerpt_id}] {e.text}" for e in self.excerpts)

    def __len__(self) -> int:
        return len(self.excerpts)


def text_digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _line_spans(text: str) -> list[tuple[int, int]]:
    """Offsets of each non-empty line, trimmed of surrounding whitespace."""
    spans: list[tuple[int, int]] = []
    offset = 0
    for raw in text.split("\n"):
        stripped = raw.strip()
        if stripped:
            lead = len(raw) - len(raw.lstrip())
            spans.append((offset + lead, offset + lead + len(stripped)))
        offset += len(raw) + 1
    return spans


def build_catalog(text: str) -> Catalog:
    """Cut ``text`` into numbered excerpts that each occur exactly once.

    Deterministic: same text, same catalogue, always. An excerpt that cannot be made unique is
    left out rather than included with a rule about which occurrence it means — an ambiguous
    entry would be exactly the failure this design exists to prevent.
    """
    if not text:
        return Catalog(digest=text_digest(text), excerpts=())

    spans = _line_spans(text)
    seen: set[str] = set()
    chosen: list[tuple[int, int, str]] = []

    # A line that identifies one place needs no neighbours. A line that does not needs a
    # neighbour that does, or the window does not say WHICH repeat is meant: "5.00\n5.00" is
    # unique as a string while pointing at both occurrences equally.
    #
    # `identifies_one_place` is the same predicate the resolver applies, overlaps and Unicode
    # twins included. Using `text.count(...) == 1` here would offer entries the resolver then
    # refuses — an inconsistency, not a shortcut.
    line_text = [text[start:end] for start, end in spans]
    unique_lines = {line for line in line_text if identifies_one_place(text, line)}

    for width in WINDOW_SIZE:
        for index in range(len(spans) - width + 1):
            window_lines = line_text[index : index + width]
            if width > 1 and not any(line in unique_lines for line in window_lines):
                continue
            start = spans[index][0]
            end = spans[index + width - 1][1]
            excerpt = text[start:end]
            if not excerpt or len(excerpt) > MAX_EXCERPT_CHARS:
                continue
            if excerpt in seen:
                continue
            # The rule, and the whole point: one occurrence, or not offered at all — with the
            # same overlap- and Unicode-aware count the resolver will use.
            if not identifies_one_place(text, excerpt):
                continue
            seen.add(excerpt)
            chosen.append((start, end, excerpt))

    # Shortest first so the budget buys the most citable spans; ties broken by position, so the
    # result is deterministic without ever consulting an expectation.
    chosen.sort(key=lambda item: (len(item[2]), item[0]))
    selected: list[tuple[int, int, str]] = []
    used = 0
    for item in chosen:
        if len(selected) >= MAX_EXCERPTS:
            break
        if used + len(item[2]) > MAX_RENDER_CHARS:
            continue
        selected.append(item)
        used += len(item[2])

    selected.sort(key=lambda item: (item[0], item[1]))
    excerpts = tuple(
        Excerpt(excerpt_id=number, start=start, end=end, text=excerpt)
        for number, (start, end, excerpt) in enumerate(selected, start=1)
    )
    return Catalog(digest=text_digest(text), excerpts=excerpts)
