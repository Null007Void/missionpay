"""Drive an authorized immutable intent to a PayPal sandbox payout, once.

The whole point of this module is the asymmetry between two kinds of failure:

* **Before dispatch** — configuration is wrong, a cap is exceeded, or a bearer
  token cannot be obtained. No byte of a payout request has been written to a
  socket. The operation stays ``PREPARED`` and may legitimately be attempted
  later. This is reported as :class:`DispatchRefused`, never as ``UNCERTAIN``.
* **After dispatch** — the ``SENDING`` state is already durable, so a POST may or
  may not have reached PayPal. That is unresolved, not failed. The operation
  becomes ``UNCERTAIN`` and this module will never send a second payout for it.

The ordering is the safety property: ``begin_dispatch`` commits ``SENDING``
*before* the adapter is called, so a crash between the two leaves a state that
recovery correctly treats as ambiguous rather than clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .domain import PaymentIntent
from .store import DispatchBlocked, MissionPayStore

TokenProvider = Callable[[], str]


class DispatchRefused(RuntimeError):
    """Refused before any payout request could be transmitted. State unchanged."""


class AcknowledgementNotDurable(RuntimeError):
    """A payout was accepted by the provider but its batch id could not be recorded."""

    def __init__(self, operation_key: str, payout_batch_id: str) -> None:
        super().__init__("PAYOUT_ACCEPTED_BUT_UNRECORDED")
        self.operation_key = operation_key
        self.payout_batch_id = payout_batch_id


@dataclass(frozen=True, slots=True)
class DispatchOutcome:
    operation_key: str
    state: str
    dispatched: bool
    payout_batch_id: str | None
    detail: str | None


class PayoutAdapter(Protocol):
    def create_payout(self, intent: PaymentIntent, *, access_token: str) -> dict[str, str]: ...


class PayoutExecutor:
    """Single-shot dispatcher. One intent, at most one payout POST, ever."""

    def __init__(
        self,
        store: MissionPayStore,
        adapter: PayoutAdapter,
        *,
        max_payout_cents: int | None = None,
        expected_recipient: str | None = None,
    ) -> None:
        if max_payout_cents is not None and max_payout_cents <= 0:
            raise ValueError("max_payout_cents must be positive when set")
        self.store = store
        self.adapter = adapter
        self.max_payout_cents = max_payout_cents
        #: The only payee a genuine dispatch may pay. An intent authorized in mock
        #: mode, or under an older configuration, can name any address; without this
        #: check it would be dispatched as-is.
        self.expected_recipient = expected_recipient

    # -- preconditions ---------------------------------------------------

    def preflight(self, operation_key: str) -> PaymentIntent:
        """Validate everything that can be validated without touching the network."""
        intent = self.store.load_intent(operation_key)
        if self.max_payout_cents is not None and intent.amount.cents > self.max_payout_cents:
            raise DispatchRefused("PAYOUT_CAP_EXCEEDED")
        if self.expected_recipient is not None and intent.recipient != self.expected_recipient:
            # Never substituted, never "corrected": the approved payee and the
            # configured payee disagree, so the operation cannot proceed.
            raise DispatchRefused("SANDBOX_RECIPIENT_MISMATCH")

        # Authority is re-verified here, not only when the intent was created. An
        # intent can outlive the approval it came from: it may predate the
        # interpretation binding entirely (a migrated row), or a materially different
        # interpretation may have been accepted afterwards. Either way the reviewer's
        # output is no longer the accepted one and the dispatch must stop.
        self._verify_approval_binding(intent)

        state = self.store.operation_state(operation_key)
        if state == "UNCERTAIN":
            raise DispatchRefused("UNCERTAIN_REQUIRES_RECONCILIATION")
        if state != "PREPARED":
            raise DispatchRefused(f"DISPATCH_STATE_NOT_PREPARED:{state}")
        return intent

    def _verify_approval_binding(self, intent: PaymentIntent) -> None:
        binding = self.store.authorization_binding(intent.operation_key)

        if not binding.get("interpretation_id") or not binding.get("interpretation_output_sha256"):
            # A legacy or migrated authorization with no reviewed output. This is not
            # repaired in place: the reviewer never saw an interpretation, so the
            # operation is quarantined until the claim is re-reviewed.
            raise DispatchRefused("AUTHORIZATION_INTERPRETATION_UNBOUND")

        if binding["claim_id"] != intent.claim_id or binding["claim_revision"] != intent.claim_revision:
            raise DispatchRefused("AUTHORIZATION_INTENT_MISMATCH")
        if binding["evidence_sha256"] != intent.evidence_sha256:
            raise DispatchRefused("AUTHORIZATION_INTENT_MISMATCH")
        if binding["fund_id"] != intent.fund_id or binding["policy_version"] != intent.policy_version:
            raise DispatchRefused("AUTHORIZATION_INTENT_MISMATCH")
        if binding["recipient"] != intent.recipient:
            raise DispatchRefused("AUTHORIZATION_INTENT_MISMATCH")
        if binding["currency"] != intent.amount.currency or binding["amount_cents"] != intent.amount.cents:
            raise DispatchRefused("AUTHORIZATION_INTENT_MISMATCH")

        if not self.store.accepted_interpretation_matches(
            intent.claim_id,
            intent.claim_revision,
            intent.evidence_sha256,
            binding["interpretation_id"],
            binding["interpretation_output_sha256"],
        ):
            raise DispatchRefused("AUTHORIZATION_INTERPRETATION_STALE")

    # -- dispatch --------------------------------------------------------

    def execute(
        self,
        operation_key: str,
        *,
        token_provider: TokenProvider,
        owner_id: str,
    ) -> DispatchOutcome:
        intent = self.preflight(operation_key)

        # A token failure proves no payout request was sent, so this is a
        # pre-dispatch refusal and the operation remains PREPARED.
        try:
            access_token = token_provider()
        except Exception as exc:  # noqa: BLE001 - classified, then re-raised cleanly
            raise DispatchRefused(f"PAYPAL_TOKEN_UNAVAILABLE:{type(exc).__name__}") from exc
        if not access_token:
            raise DispatchRefused("PAYPAL_TOKEN_UNAVAILABLE:EMPTY")

        # From here on a request may reach the provider. SENDING is durable first.
        try:
            self.store.begin_dispatch(operation_key, owner_id)
        except DispatchBlocked as exc:
            raise DispatchRefused(f"DISPATCH_BLOCKED:{exc}") from exc

        try:
            result = self.adapter.create_payout(intent, access_token=access_token)
            payout_batch_id = result.get("payout_batch_id")
        except Exception as exc:  # noqa: BLE001 - ambiguity is the point
            detail = f"POST_OUTCOME_UNKNOWN:{type(exc).__name__}"
            self._mark_uncertain(operation_key, owner_id, detail)
            return DispatchOutcome(
                operation_key=operation_key,
                state="UNCERTAIN",
                dispatched=True,
                payout_batch_id=None,
                detail=detail,
            )

        if not isinstance(payout_batch_id, str) or not payout_batch_id:
            detail = "PROVIDER_RESPONSE_WITHOUT_BATCH_ID"
            self._mark_uncertain(operation_key, owner_id, detail)
            return DispatchOutcome(
                operation_key=operation_key,
                state="UNCERTAIN",
                dispatched=True,
                payout_batch_id=None,
                detail=detail,
            )

        try:
            self.store.record_payout_acknowledgement(operation_key, payout_batch_id)
        except Exception as exc:  # noqa: BLE001 - must not lose the batch id
            # The provider accepted the payout. Keep the authoritative identifier in
            # the durable detail so an operator can reconcile rather than guess, and
            # leave the operation unresolved.
            self._mark_uncertain(
                operation_key,
                owner_id,
                f"ACK_NOT_RECORDED batch={payout_batch_id}",
            )
            raise AcknowledgementNotDurable(operation_key, payout_batch_id) from exc

        return DispatchOutcome(
            operation_key=operation_key,
            state="ACKNOWLEDGED",
            dispatched=True,
            payout_batch_id=payout_batch_id,
            detail=None,
        )

    def _mark_uncertain(self, operation_key: str, owner_id: str, detail: str) -> None:
        try:
            self.store.mark_dispatch_uncertain(operation_key, owner_id, detail)
        except DispatchBlocked:
            # Already moved on (a concurrent observer) - do not overwrite history.
            pass
