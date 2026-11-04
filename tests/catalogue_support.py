"""Helpers for tests that speak the excerpt-catalogue contract (v7+).

The model no longer returns text or offsets: it returns ``excerpt_id``. Tests still want to
express intent as "cite the total line" or "cite this snippet", so these helpers translate a
snippet into the catalogue number MissionPay would have offered for it.

They never invent an id. If the snippet is not carried by exactly one excerpt, the helper
returns ``None`` and the caller has to decide what that means — which is usually the point of
the test.
"""

from __future__ import annotations

import json

from missionpay.catalog import build_catalog


def catalogue(text: str):
    return build_catalog(text)


def id_for(text: str, snippet: str) -> int | None:
    """The number of the excerpt that carries ``snippet``, or None if no excerpt does.

    If more than one excerpt carries it, the shortest is returned — the tightest citation —
    but callers that care about uniqueness should assert on the catalogue directly.
    """
    matches = [e for e in catalogue(text).excerpts if snippet in e.text]
    if not matches:
        return None
    return min(matches, key=lambda e: (len(e.text), e.excerpt_id)).excerpt_id


def excerpt_text(text: str, excerpt_id: int) -> str:
    for entry in catalogue(text).excerpts:
        if entry.excerpt_id == excerpt_id:
            return entry.text
    raise AssertionError(f"no excerpt {excerpt_id} in this text")


def fact(
    text: str,
    name: str,
    value: str,
    status: str = "observed",
    snippet: str | None = None,
    excerpt_id: object | None = None,
) -> dict[str, object]:
    """One fact in the shape the schema requires."""
    if excerpt_id is None:
        assert snippet is not None, "give a snippet or an explicit excerpt_id"
        excerpt_id = id_for(text, snippet)
        assert excerpt_id is not None, f"no excerpt carries {snippet!r}"
    return {
        "name": name,
        "value": value,
        "epistemic_status": status,
        "excerpt_id": excerpt_id,
    }


class PayloadModel:
    """A model double that returns a fixed payload, in the current schema."""

    def __init__(self, payload: dict, model_id: str = "test-fixture/v1") -> None:
        self.payload = payload
        self.model_id = model_id

    def complete_json(self, system: str, user: str) -> str:  # noqa: ARG002
        return json.dumps(self.payload)


def answer(*facts: dict, missing: list[str] | None = None) -> dict[str, object]:
    return {
        "facts": list(facts),
        "missing": missing if missing is not None else ["independent proof of attendance"],
    }
