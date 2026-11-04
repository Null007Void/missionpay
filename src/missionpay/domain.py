"""Immutable value objects for MissionPay's deterministic core."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import re

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _nonempty(name: str, value: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and trimmed")


@dataclass(frozen=True, slots=True)
class Money:
    currency: str
    cents: int

    def __post_init__(self) -> None:
        if self.currency != "USD":
            raise ValueError("MissionPay R2 supports USD only")
        if isinstance(self.cents, bool) or not isinstance(self.cents, int):
            raise TypeError("cents must be an integer")
        if self.cents <= 0:
            raise ValueError("cents must be positive")


@dataclass(frozen=True, slots=True)
class FundBalance:
    """A restricted-fund *available balance*, which may legitimately reach zero.

    A payout amount stays a strictly positive ``Money``; only the remaining
    balance can be exactly exhausted. Exhaustion is a normal, loadable state:
    zero available means the next positive claim is refused, never that the
    policy record is unreadable.
    """

    currency: str
    cents: int

    def __post_init__(self) -> None:
        if self.currency != "USD":
            raise ValueError("MissionPay R2 supports USD only")
        if isinstance(self.cents, bool) or not isinstance(self.cents, int):
            raise TypeError("cents must be an integer")
        if self.cents < 0:
            raise ValueError("available balance cannot be negative")

    @property
    def is_exhausted(self) -> bool:
        return self.cents == 0


@dataclass(frozen=True, slots=True)
class EvidenceIdentity:
    sha256: str
    media_type: str
    original_name: str

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        _nonempty("media_type", self.media_type)
        _nonempty("original_name", self.original_name)


@dataclass(frozen=True, slots=True)
class Activity:
    activity_id: str
    activity_code: str
    occurred_on: date

    def __post_init__(self) -> None:
        _nonempty("activity_id", self.activity_id)
        if not _CODE.fullmatch(self.activity_code):
            raise ValueError("activity_code must be uppercase snake case")


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    volunteer_id: str
    activity: Activity
    evidence: EvidenceIdentity
    requested: Money
    recipient: str
    expense_category: str | None

    def __post_init__(self) -> None:
        _nonempty("claim_id", self.claim_id)
        _nonempty("volunteer_id", self.volunteer_id)
        _nonempty("recipient", self.recipient)
        if self.expense_category is not None and not _CODE.fullmatch(self.expense_category):
            raise ValueError("expense_category must be uppercase snake case")


@dataclass(frozen=True, slots=True)
class FundPolicy:
    fund_id: str
    version: str
    allowed_activity_codes: frozenset[str]
    allowed_expense_categories: frozenset[str]
    starts_on: date
    ends_on: date
    available: FundBalance | Money
    receipt_required: bool

    def __post_init__(self) -> None:
        if isinstance(self.available, Money):
            object.__setattr__(
                self, "available", FundBalance(self.available.currency, self.available.cents)
            )
        elif not isinstance(self.available, FundBalance):
            raise TypeError("available must be Money or FundBalance")
        _nonempty("fund_id", self.fund_id)
        _nonempty("version", self.version)
        if self.starts_on > self.ends_on:
            raise ValueError("policy starts_on must not be after ends_on")
        if not self.allowed_activity_codes or not all(
            _CODE.fullmatch(code) for code in self.allowed_activity_codes
        ):
            raise ValueError("allowed_activity_codes must contain valid codes")
        if not self.allowed_expense_categories or not all(
            _CODE.fullmatch(code) for code in self.allowed_expense_categories
        ):
            raise ValueError("allowed_expense_categories must contain valid codes")
        if self.available.currency != "USD":
            raise ValueError("fund currency must be USD")


class PolicyOutcome(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    claim_id: str
    fund_id: str
    policy_version: str
    outcome: PolicyOutcome
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.reason_codes:
            raise ValueError("policy decision requires at least one reason code")


@dataclass(frozen=True, slots=True)
class HumanAuthorization:
    authorization_id: str
    reviewer_id: str
    reviewer_role: str
    authenticated: bool
    claim_id: str
    claim_revision: int
    evidence_sha256: str
    policy_version: str
    fund_id: str
    recipient: str
    amount: Money

    def __post_init__(self) -> None:
        for name in (
            "authorization_id",
            "reviewer_id",
            "claim_id",
            "policy_version",
            "fund_id",
            "recipient",
        ):
            _nonempty(name, getattr(self, name))
        if self.reviewer_role != "REIMBURSEMENT_REVIEWER":
            raise ValueError("reviewer_role is not authorized to approve reimbursements")
        if not _SHA256.fullmatch(self.evidence_sha256):
            raise ValueError("authorization evidence digest is invalid")
        if isinstance(self.claim_revision, bool) or self.claim_revision < 1:
            raise ValueError("claim_revision must be a positive integer")


@dataclass(frozen=True, slots=True)
class PaymentIntent:
    operation_key: str
    claim_id: str
    claim_revision: int
    evidence_sha256: str
    policy_version: str
    fund_id: str
    recipient: str
    amount: Money
    authorization_id: str

    def __post_init__(self) -> None:
        if not self.operation_key.startswith("missionpay:"):
            raise ValueError("payment operation key must be MissionPay namespaced")
        if isinstance(self.claim_revision, bool) or self.claim_revision < 1:
            raise ValueError("claim_revision must be a positive integer")


def _aware(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


class ExecutionState(str, Enum):
    PREPARED = "prepared"
    SUBMITTED = "submitted"
    UNCERTAIN = "uncertain"
    OBSERVED = "observed"
    RECONCILED = "reconciled"


@dataclass(frozen=True, slots=True)
class ExecutionAttempt:
    attempt_id: str
    operation_key: str
    request_sha256: str
    state: ExecutionState
    created_at: datetime

    def __post_init__(self) -> None:
        _nonempty("attempt_id", self.attempt_id)
        if not self.operation_key.startswith("missionpay:"):
            raise ValueError("execution operation key must be MissionPay namespaced")
        if not _SHA256.fullmatch(self.request_sha256):
            raise ValueError("request_sha256 is invalid")
        _aware("created_at", self.created_at)


class ProviderItemStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PENDING = "PENDING"
    UNCLAIMED = "UNCLAIMED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    RETURNED = "RETURNED"
    ONHOLD = "ONHOLD"
    REFUNDED = "REFUNDED"
    REVERSED = "REVERSED"


@dataclass(frozen=True, slots=True)
class ProviderObservation:
    observation_id: str
    attempt_id: str
    provider: str
    payout_batch_id: str
    payout_item_id: str
    status: ProviderItemStatus
    recipient: str
    amount: Money
    observed_at: datetime
    raw_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "observation_id",
            "attempt_id",
            "payout_batch_id",
            "payout_item_id",
            "recipient",
        ):
            _nonempty(name, getattr(self, name))
        if self.provider != "PAYPAL_SANDBOX":
            raise ValueError("provider must be PayPal sandbox")
        _aware("observed_at", self.observed_at)
        if not _SHA256.fullmatch(self.raw_sha256):
            raise ValueError("raw_sha256 is invalid")


class ReconciliationOutcome(str, Enum):
    MATCHED_COMPLETED = "matched_completed"
    PENDING = "pending"
    FAILED = "failed"
    MISMATCH = "mismatch"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class Reconciliation:
    operation_key: str
    observation_id: str
    outcome: ReconciliationOutcome
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.operation_key.startswith("missionpay:"):
            raise ValueError("reconciliation operation key must be MissionPay namespaced")
        _nonempty("observation_id", self.observation_id)
        if not self.reason_codes:
            raise ValueError("reconciliation requires at least one reason code")


@dataclass(frozen=True, slots=True)
class AuditEvent:
    event_id: str
    sequence: int
    aggregate_id: str
    event_type: str
    payload_sha256: str
    occurred_at: datetime

    def __post_init__(self) -> None:
        _nonempty("event_id", self.event_id)
        _nonempty("aggregate_id", self.aggregate_id)
        if not _CODE.fullmatch(self.event_type):
            raise ValueError("event_type must be uppercase snake case")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("sequence must be a positive integer")
        if not _SHA256.fullmatch(self.payload_sha256):
            raise ValueError("payload_sha256 is invalid")
        _aware("occurred_at", self.occurred_at)
