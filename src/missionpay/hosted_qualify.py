"""One-shot hosted qualification run.

The product itself is not single-use: a volunteer can interpret any number of claims in
fixture or local mode, and a reviewer can approve as many claims as the fund allows.
What must be single-use is *this* — one paid request whose purpose is to prove the hosted
path works end to end. The authorisation reference is the unit of that guarantee.

The guarantee is durable, not a process flag:

1. the attempt row is inserted and committed **before** the socket is opened, so a crash,
   a timeout or a restart leaves the row behind;
2. the row's primary key is the authorisation reference, so a second run under the same
   reference is refused by the database, not by a variable in memory;
3. the row pins what was approved — claim revision, evidence digest, model, configuration
   digest and both token ceilings — and a trigger refuses an update that would rewrite
   any of them;
4. only the outcome may be filled in, and only once, from ``RECORDED``.

A new attempt therefore needs a new authorisation reference from the owner. That is the
whole point: the second send is a decision, never an accident.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path

from .ai import (
    FACT_NAMES,
    MAX_EVIDENCE_TEXT,
    PROMPT_VERSION,
    ModelInterpretationError,
    ReceiptInterpretation,
    interpretation_prompt,
    interpret_receipt_text,
)
from .domain import EvidenceIdentity
from .config import AiMode, ConfigurationError, RuntimeConfig
from .evidence import EvidenceStore, default_evidence_root
from .hosted import HostedJsonModel, HostedModelError
from .store import MissionPayStore, PersistenceConflict, clean_diagnostic


#: An amount, optionally with a currency prefix or suffix. Parsed before any case
#: folding: lower-casing the text first is what made ``USD 5.00`` stop matching.
_MONEY_PATTERN = re.compile(
    r"^(?:(?P<prefix>[A-Za-z]{3})\s*)?\$?\s*"
    r"(?P<amount>\d{1,9}(?:\.\d{1,2})?)"
    r"(?:\s*(?P<suffix>[A-Za-z]{3}))?$"
)


class QualificationRefused(RuntimeError):
    """A refusal that costs nothing, raised before anything is sent."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class QualificationPlan:
    """What has been approved, pinned as one immutable bundle."""

    authorization_id: str
    claim_id: str
    revision: int
    evidence_sha256: str
    expectation_sha256: str
    model_id: str
    config_sha256: str
    max_output_tokens: int
    max_total_tokens: int
    timeout_seconds: int
    late_response_seconds: int
    estimate: dict[str, int]
    evidence_chars: int
    prompt_chars: int

    def as_dict(self) -> dict[str, object]:
        return {
            "authorization_id": self.authorization_id,
            "claim_id": self.claim_id,
            "revision": self.revision,
            "evidence_sha256": self.evidence_sha256,
            "expectation_sha256": self.expectation_sha256,
            "model_id": self.model_id,
            "config_sha256": self.config_sha256,
            "max_output_tokens": self.max_output_tokens,
            "max_total_tokens": self.max_total_tokens,
            "timeout_seconds": self.timeout_seconds,
            "late_response_seconds": self.late_response_seconds,
            "evidence_chars": self.evidence_chars,
            "prompt_chars": self.prompt_chars,
            "token_estimate": self.estimate,
            "estimate_is_a_measurement": False,
        }


def _qualification_digest(
    *,
    model_id: str,
    evidence_sha256: str,
    expectation_sha256: str,
    max_output_tokens: int,
    max_total_tokens: int,
    timeout_seconds: int,
    late_response_seconds: int,
) -> str:
    """Identity of the approved request.

    Everything that changes what the run may cost, how long it may take, or what counts
    as a pass is inside this digest: both token ceilings, the socket timeout, the
    late-response threshold, and the expectation being judged against. Widening any of
    them produces a different identity, so a stored attempt cannot be read as approving a
    wider request than the one it recorded.
    """
    return sha256(
        json.dumps(
            {
                "ai_mode": AiMode.HOSTED.value,
                "provider": "openai",
                "model_id": model_id,
                "prompt_version": PROMPT_VERSION,
                "evidence_sha256": evidence_sha256,
                "expectation_sha256": expectation_sha256,
                "max_output_tokens": max_output_tokens,
                "max_total_tokens": max_total_tokens,
                "timeout_seconds": timeout_seconds,
                "late_response_seconds": late_response_seconds,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_model(config: RuntimeConfig) -> HostedJsonModel:
    if config.ai_mode is not AiMode.HOSTED:
        raise QualificationRefused("HOSTED_MODE_REQUIRED")
    try:
        return HostedJsonModel(
            base_url=config.hosted_base_url,
            model_id=config.hosted_model or "",
            api_key=config.hosted_api_key or "",
            max_output_tokens=config.hosted_max_output_tokens or 0,
            max_total_tokens=config.hosted_max_total_tokens or 0,
            timeout_seconds=config.hosted_timeout_seconds,
            late_response_seconds=config.hosted_late_response_seconds,
        )
    except HostedModelError as exc:
        raise QualificationRefused(exc.code) from exc


def build_plan(
    store: MissionPayStore,
    config: RuntimeConfig,
    *,
    authorization_id: str,
    claim_id: str,
    revision: int | None,
    expectation: dict[str, object],
    model: HostedJsonModel | None = None,
) -> tuple[QualificationPlan, str, EvidenceIdentity]:
    """Pin the request without sending it: plan, evidence text and evidence identity.

    The expectation is an argument, not an afterthought. It is validated and its digest is
    pinned here, before anything can be sent, so a missing or malformed expectation stops
    the run while it still costs nothing.
    """
    if not authorization_id or not authorization_id.strip():
        raise QualificationRefused("AUTHORIZATION_REFERENCE_REQUIRED")
    expectation_sha256 = expectation_digest(expectation)
    model = model or build_model(config)
    try:
        snapshot = store.claim_by_id(claim_id, revision)
    except PersistenceConflict as exc:
        raise QualificationRefused("CLAIM_NOT_FOUND") from exc
    claim = snapshot["claim"]
    revision_value = int(snapshot["revision"])

    text = EvidenceStore(default_evidence_root(store.path)).read_text(claim.evidence.sha256)
    if text is None:
        raise QualificationRefused("EVIDENCE_TEXT_UNAVAILABLE")
    if len(text) > MAX_EVIDENCE_TEXT:
        raise QualificationRefused("EVIDENCE_TEXT_SIZE_INVALID")

    # The expectation must describe *this* receipt. A verdict written against other
    # evidence would be meaningless even if every value happened to match.
    declared = str((expectation.get("fixture") or {}).get("sha256") or "")
    if declared != claim.evidence.sha256:
        raise QualificationRefused("EXPECTATION_RECEIPT_MISMATCH")
    validate_expectation_against(expectation, text)

    system, user = interpretation_prompt(text)
    estimate = model.preview(system, user)
    if estimate["upper_bound_total_tokens"] > model.max_total_tokens:
        # Refused before the attempt is recorded, because nothing was sent and the
        # authorisation has not been consumed. Checked against the byte-length preflight,
        # which is a conservative assumption rather than a proved bound: it holds only if
        # the provider tokenizer emits no more tokens than the payload has bytes.
        raise QualificationRefused("HOSTED_REQUEST_OVER_BUDGET")

    plan = QualificationPlan(
        authorization_id=authorization_id.strip(),
        claim_id=claim_id,
        revision=revision_value,
        evidence_sha256=claim.evidence.sha256,
        expectation_sha256=expectation_sha256,
        model_id=model.qualified_model_id,
        config_sha256=_qualification_digest(
            model_id=model.qualified_model_id,
            evidence_sha256=claim.evidence.sha256,
            expectation_sha256=expectation_sha256,
            max_output_tokens=model.max_output_tokens,
            max_total_tokens=model.max_total_tokens,
            timeout_seconds=model.timeout_seconds,
            late_response_seconds=model.late_response_seconds,
        ),
        max_output_tokens=model.max_output_tokens,
        max_total_tokens=model.max_total_tokens,
        timeout_seconds=model.timeout_seconds,
        late_response_seconds=model.late_response_seconds,
        estimate=estimate,
        evidence_chars=len(text),
        prompt_chars=len(system) + len(user),
    )
    return plan, text, claim.evidence


def _terminal_state(code: str) -> str:
    """Map a refusal code to what the record says happened.

    ``UNCERTAIN`` is not a euphemism: a transport error means the request may have been
    received and may have been charged. It is recorded as uncertain so nobody reads it
    later as "nothing happened, try again".
    """
    if code in {"HOSTED_TIMEOUT", "HOSTED_DEADLINE_EXCEEDED"}:
        return "TIMEOUT"
    if code in {"HOSTED_NETWORK_ERROR", "HOSTED_TRANSPORT_ERROR"}:
        return "UNCERTAIN"
    return "FAILED"


def qualify(
    store: MissionPayStore,
    config: RuntimeConfig,
    *,
    authorization_id: str,
    claim_id: str,
    revision: int | None = None,
    expectation: dict[str, object],
    model: HostedJsonModel | None = None,
) -> dict[str, object]:
    """Record the attempt, make exactly one request, then judge it.

    Three outcomes are kept apart on purpose:

    * ``QUALIFIED`` — the provider completed and the answer matched the frozen expectation;
    * ``VERDICT_FAILED`` — the provider completed and the answer did **not** match. The
      request was paid for, but nothing was qualified;
    * ``FAILED`` / ``TIMEOUT`` / ``UNCERTAIN`` — no answer to judge.

    A completed request is never reported as a successful qualification on its own.
    """
    plan, text, evidence = build_plan(
        store,
        config,
        authorization_id=authorization_id,
        claim_id=claim_id,
        revision=revision,
        expectation=expectation,
        model=model,
    )
    model = model or build_model(config)

    # -- the durable gate, before the request exists ----------------------
    try:
        store.record_hosted_attempt(
            authorization_id=plan.authorization_id,
            claim_id=plan.claim_id,
            claim_revision=plan.revision,
            evidence_sha256=plan.evidence_sha256,
            expectation_sha256=plan.expectation_sha256,
            model_id=plan.model_id,
            config_sha256=plan.config_sha256,
            max_output_tokens=plan.max_output_tokens,
            max_total_tokens=plan.max_total_tokens,
        )
    except PersistenceConflict:
        existing = store.hosted_attempt(plan.authorization_id) or {}
        return {
            "status": "ALREADY_ATTEMPTED",
            "sent": False,
            "plan": plan.as_dict(),
            "attempt": existing,
            "detail": (
                "This authorisation has already been used for one request. A second "
                "attempt needs a new authorisation reference from the owner."
            ),
        }

    # -- exactly one send -------------------------------------------------
    try:
        result = interpret_receipt_text(evidence=evidence, text=text, model=model)
    except (ModelInterpretationError, HostedModelError) as exc:
        code = getattr(exc, "code", None) or str(exc)
        state = _terminal_state(code)
        # Sanitised exactly once, here at the boundary. The same object goes to the
        # journal, the outcome and the CLI; nothing downstream sees the raw attribute,
        # which any exception could carry.
        diagnostic = clean_diagnostic(getattr(exc, "diagnostic", None))
        store.finish_hosted_attempt(
            plan.authorization_id,
            state=state,
            code=code,
            usage=model.last_usage,
            diagnostic=diagnostic,
        )
        return {
            "status": state,
            "sent": True,
            "code": code,
            "diagnostic": diagnostic,
            "plan": plan.as_dict(),
            "usage": model.last_usage,
            "elapsed_seconds": model.last_elapsed_seconds,
            "attempt": store.hosted_attempt(plan.authorization_id),
        }

    interpretation = _interpretation_dict(result)
    verdict = compare(interpretation, expectation, text)
    store.finish_hosted_attempt(
        plan.authorization_id, state="SUCCEEDED", usage=model.last_usage
    )
    return {
        # Provider completion and qualification are different facts, so they get different
        # names. The stored attempt says SUCCEEDED (the provider answered); this status says
        # whether the answer qualified.
        "status": "QUALIFIED" if verdict["passed"] else "VERDICT_FAILED",
        "provider_completed": True,
        "sent": True,
        "plan": plan.as_dict(),
        "usage": model.last_usage,
        "elapsed_seconds": model.last_elapsed_seconds,
        "interpretation": interpretation,
        "verdict": verdict,
        "attempt": store.hosted_attempt(plan.authorization_id),
    }


def _interpretation_dict(result: ReceiptInterpretation) -> dict[str, object]:
    return {
        "facts": [
            {
                "name": fact.name,
                "value": fact.value,
                "epistemic_status": fact.epistemic_status.value,
                "start": fact.citation.start,
                "end": fact.citation.end,
                "quote": fact.citation.quote,
                # Provenance travels with the outcome: which catalogue entry the model chose,
                # and whether the span was resolved here (it always is, under v7).
                "excerpt_id": fact.citation.excerpt_id,
                "offsets_derived": fact.offsets_derived,
            }
            for fact in result.facts
        ],
        "missing": list(result.missing),
    }


# -- the frozen expectation -------------------------------------------------

#: Statuses the contract allows. Kept here so an expectation that names something else is
#: refused before a request is paid for rather than rejected by the verdict afterwards.
_VALID_STATUSES = ("observed", "inference", "uncertain")


def expectation_digest(expectation: dict[str, object]) -> str:
    """Canonical digest of the expectation, pinned in the plan and the attempt row."""
    return sha256(
        json.dumps(expectation, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_expectation(target: str | Path) -> dict[str, object]:
    """Load and validate an expectation. Everything here happens before any send.

    A missing, empty, malformed or incomplete file is a refusal with its own code. The
    previous version returned ``{}`` for an empty file, and an empty expectation compared
    equal to everything, which made the verdict meaningless.
    """
    path = Path(target)
    if path.is_dir():
        path = path / "expected.json"
    if not path.exists():
        raise QualificationRefused("EXPECTATION_MISSING")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise QualificationRefused("EXPECTATION_UNREADABLE") from exc
    if not raw.strip():
        raise QualificationRefused("EXPECTATION_EMPTY")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QualificationRefused("EXPECTATION_INVALID") from exc
    if not isinstance(data, dict) or not data:
        raise QualificationRefused("EXPECTATION_INVALID")

    fixture = data.get("fixture")
    if not isinstance(fixture, dict):
        raise QualificationRefused("EXPECTATION_INCOMPLETE")
    declared = str(fixture.get("sha256") or "")
    if len(declared) != 64 or any(char not in "0123456789abcdef" for char in declared):
        raise QualificationRefused("EXPECTATION_INCOMPLETE")

    facts = data.get("expected_facts")
    if not isinstance(facts, list) or not facts:
        raise QualificationRefused("EXPECTATION_EMPTY")
    seen: set[str] = set()
    for fact in facts:
        if not isinstance(fact, dict):
            raise QualificationRefused("EXPECTATION_INCOMPLETE")
        name = fact.get("name")
        if not isinstance(name, str) or not name:
            raise QualificationRefused("EXPECTATION_INCOMPLETE")
        if name in seen:
            raise QualificationRefused("EXPECTATION_INCOMPLETE")
        seen.add(name)
        if name not in FACT_NAMES:
            raise QualificationRefused("EXPECTATION_UNKNOWN_FACT")
        if not isinstance(fact.get("value"), str) or not str(fact.get("value")).strip():
            raise QualificationRefused("EXPECTATION_INCOMPLETE")
        if fact.get("epistemic_status") not in _VALID_STATUSES:
            raise QualificationRefused("EXPECTATION_INCOMPLETE")
        if not isinstance(fact.get("anchor"), str) or not str(fact.get("anchor")):
            raise QualificationRefused("EXPECTATION_INCOMPLETE")
        for key in ("canonical_start", "canonical_end"):
            if key in fact and (
                isinstance(fact[key], bool) or not isinstance(fact[key], int)
            ):
                raise QualificationRefused("EXPECTATION_INCOMPLETE")
    return data


def validate_expectation_against(expectation: dict[str, object], text: str) -> None:
    """The expectation must describe this receipt: digest, size and unique anchors."""
    fixture = expectation.get("fixture") or {}
    if str(fixture.get("sha256")) != sha256(text.encode("utf-8")).hexdigest():
        raise QualificationRefused("EXPECTATION_RECEIPT_MISMATCH")
    declared_chars = fixture.get("characters")
    if isinstance(declared_chars, int) and declared_chars != len(text):
        raise QualificationRefused("EXPECTATION_RECEIPT_MISMATCH")
    for fact in expectation.get("expected_facts") or []:
        anchor = str(fact.get("anchor"))
        if text.count(anchor) != 1:
            # An anchor that is absent, or appears twice, makes the citation rule
            # ambiguous; refuse before spending rather than argue about it afterwards.
            raise QualificationRefused("EXPECTATION_ANCHOR_NOT_UNIQUE")


def compare(
    interpretation: dict[str, object], expectation: dict[str, object], text: str
) -> dict[str, object]:
    """Compare a live answer against the expectation frozen before it was made.

    The expectation is read, never written. Nothing here can adjust it to fit an answer:
    a mismatch is reported as a mismatch and the run stops.
    """
    facts = interpretation.get("facts") or []
    by_name: dict[str, list[dict[str, object]]] = {}
    for fact in facts:
        by_name.setdefault(str(fact.get("name")), []).append(fact)

    expected = expectation.get("expected_facts") or []
    expected_names = {str(spec.get("name")) for spec in expected}
    mismatches: list[str] = []
    if not expected:
        # Defensive: load_expectation already refuses an empty expectation, and this
        # second line of defence stops a hand-built one from passing everything.
        mismatches.append("expectation: no expected facts to judge against")

    for spec in expected:
        name = str(spec.get("name"))
        anchor = str(spec.get("anchor") or "")
        candidates = by_name.get(name) or []
        if not candidates:
            mismatches.append(f"{name}: absent from the answer")
            continue
        if len(candidates) > 1:
            mismatches.append(f"{name}: stated {len(candidates)} times, expected once")
        fact = candidates[0]
        value = str(fact.get("value", ""))
        if not _same_value(value, str(spec.get("value", ""))):
            mismatches.append(f"{name}: value {value!r} is not {spec.get('value')!r}")
        # How the fact is known is part of the claim. A model that reports an inference as
        # observed has said something the receipt does not support, even with a perfect
        # citation, so the status is judged as strictly as the value.
        status = fact.get("epistemic_status")
        if status != spec.get("epistemic_status"):
            mismatches.append(
                f"{name}: status {status!r} is not {spec.get('epistemic_status')!r}"
            )

        # The receipt must actually contain the anchor once, or the expectation itself
        # has drifted from the fixture and the verdict would be meaningless.
        if anchor and text.count(anchor) != 1:
            mismatches.append(
                f"{name}: expectation anchor occurs {text.count(anchor)} times, not once"
            )
            continue

        start = fact.get("start")
        end = fact.get("end")
        quote = fact.get("quote")
        if isinstance(start, bool) or isinstance(end, bool):
            mismatches.append(f"{name}: citation offsets are not integers")
            continue
        if not isinstance(start, int) or not isinstance(end, int):
            mismatches.append(f"{name}: citation offsets missing")
            continue
        if not isinstance(quote, str):
            mismatches.append(f"{name}: quote missing")
            continue
        if not (0 <= start < end <= len(text)):
            mismatches.append(f"{name}: citation span {start}:{end} lies outside the text")
            continue
        if text[start:end] != quote:
            # The real check: the quote must be the text at the offsets, not a
            # paraphrase that happens to be nearby.
            mismatches.append(f"{name}: quote is not receipt text {start}:{end}")
            continue
        if anchor and anchor not in quote:
            mismatches.append(f"{name}: quote does not contain the anchor {anchor!r}")

    # A frozen benchmark cannot pass a fact nobody expected. Reporting these and returning
    # passed=True let an answer carry a second, contradictory total inside a clean verdict:
    # the expectation named total_paid, so a model (or an injected fact) could state
    # total_usd = "USD 9.00" and still qualify. Every unexpected fact now fails the verdict.
    unexpected = sorted(set(by_name) - expected_names)
    for name in unexpected:
        stated = ", ".join(repr(fact.get("value")) for fact in by_name[name])
        mismatches.append(
            f"{name}: stated but not in the frozen expectation (value {stated})"
        )

    missing = interpretation.get("missing")
    missing_ok = (
        isinstance(missing, list)
        and bool(missing)
        and all(isinstance(item, str) and item.strip() for item in missing)
    )
    if not missing_ok:
        # This receipt cannot establish attendance, so an answer that claims nothing is
        # missing has over-read the evidence.
        mismatches.append("missing: at least one absent fact must be stated")

    return {
        "passed": not mismatches,
        "mismatches": mismatches,
        "unexpected_facts": unexpected,
        "missing": missing,
        "citation_rule": expectation.get("citation_rule"),
        "expectation_sha256": sha256(
            json.dumps(expectation, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _same_value(observed: str, expected: str) -> bool:
    """Equality for a fact value, comparing money as money.

    The previous version case-folded before parsing and compared with ``float``, which
    broke on ``USD 5.00`` versus ``5.00`` (the currency prefix stopped matching once it
    was lower-cased) and could call two different amounts equal at a rounding boundary.
    Money is now parsed first, compared as ``Decimal``, and two explicitly different
    currencies are never equal.
    """
    left_money = _money(observed)
    right_money = _money(expected)
    if left_money is not None and right_money is not None:
        left_amount, left_currency = left_money
        right_amount, right_currency = right_money
        if left_currency and right_currency and left_currency != right_currency:
            return False
        return left_amount == right_amount
    if left_money is not None or right_money is not None:
        # One side is an amount and the other is not: not the same fact value.
        return False
    return " ".join(observed.split()).casefold() == " ".join(expected.split()).casefold()


def _money(value: str) -> tuple[Decimal, str] | None:
    stripped = value.strip()
    match = _MONEY_PATTERN.match(stripped)
    if match is None:
        return None
    prefix = (match.group("prefix") or "").upper()
    suffix = (match.group("suffix") or "").upper()
    if prefix:
        currency = prefix
    elif "$" in stripped:
        currency = "USD"
    else:
        currency = suffix
    try:
        return Decimal(match.group("amount")), currency
    except InvalidOperation:  # pragma: no cover - the pattern already constrains this
        return None
