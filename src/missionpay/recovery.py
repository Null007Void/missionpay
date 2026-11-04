"""Conservative recovery and readback rules for ambiguous PayPal payout dispatches.

Every path in this module is read-only against the provider: batch and item GETs
only. No method here constructs another payout POST, and no ambiguity is resolved
by guessing. Later observations of an already acknowledged payout append evidence
and re-derive the operation's financial state; they never overwrite history.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from hashlib import sha256
import json
from typing import Protocol
from uuid import uuid4

from .domain import (
    Money,
    PaymentIntent,
    ProviderItemStatus,
    ProviderObservation,
)
from .reconciliation import reconcile_payment
from .store import OBSERVABLE_OPERATION_STATES, MissionPayStore


class RecoveryDisposition(str, Enum):
    BLOCKED_UNKNOWN_BATCH_ID = "blocked_unknown_batch_id"
    RECONCILED = "reconciled"
    PENDING = "pending"
    SUPERSEDED = "superseded"
    CONFLICT = "conflict"


class RecoveryAdapter(Protocol):
    def get_batch(self, payout_batch_id: str, *, access_token: str) -> dict[str, object]: ...

    def get_item(self, payout_item_id: str, *, access_token: str) -> dict[str, object]: ...


class PayPalRecovery:
    def __init__(self, store: MissionPayStore, adapter: RecoveryAdapter):
        self.store = store
        self.adapter = adapter

    def recover_unknown_batch(
        self,
        operation_key: str,
        *,
        access_token: str,
    ) -> RecoveryDisposition:
        """Fail closed when a possibly accepted POST lost its batch identifier.

        PayPal's documented batch GET requires payout_batch_id. MissionPay neither
        fabricates one nor assumes sender_batch_id is searchable, and it never
        issues another payout POST from this recovery path.
        """
        del access_token  # no provider request is valid without an authoritative batch ID
        if self.store.operation_state(operation_key) != "UNCERTAIN":
            raise ValueError("UNCERTAIN_OPERATION_REQUIRED")
        return RecoveryDisposition.BLOCKED_UNKNOWN_BATCH_ID

    def recover_with_batch_id(
        self,
        operation_key: str,
        payout_batch_id: str,
        *,
        access_token: str,
    ) -> RecoveryDisposition:
        """Bind an authoritative batch to an UNCERTAIN operation and observe its item."""
        if self.store.operation_state(operation_key) != "UNCERTAIN":
            raise ValueError("UNCERTAIN_OPERATION_REQUIRED")
        intent = self.store.load_intent(operation_key)
        batch = self.adapter.get_batch(payout_batch_id, access_token=access_token)
        items, failure = self._batch_items(operation_key, batch, payout_batch_id)
        if failure is not None:
            return self._record_conflict(operation_key, failure)
        matching = self._matching_items(items, intent)
        if len(matching) != 1 or not self._item_matches_intent(matching[0], intent):
            return self._record_conflict(operation_key, "RECOVERED_BATCH_ITEM_MISMATCH")
        payout_item_id = matching[0].get("payout_item_id")
        if not isinstance(payout_item_id, str) or not payout_item_id:
            return self._record_conflict(operation_key, "RECOVERED_ITEM_ID_MISSING")

        item = self.adapter.get_item(payout_item_id, access_token=access_token)
        if item.get("payout_item_id") != payout_item_id or not self._item_matches_intent(item, intent):
            return self._record_conflict(operation_key, "RECOVERED_ITEM_DETAIL_MISMATCH")

        self.store.record_payout_acknowledgement(operation_key, payout_batch_id)
        return self._persist_item_observation(operation_key, payout_batch_id, payout_item_id, item)

    def observe_known_payout(
        self,
        operation_key: str,
        *,
        access_token: str,
    ) -> RecoveryDisposition:
        """Read an already acknowledged payout again and append the observed state.

        This is the routine asynchronous transition path (PENDING/UNCLAIMED ->
        SUCCESS, SUCCESS -> REFUNDED/REVERSED/RETURNED). It requires an authoritative
        batch binding, uses provider GETs only, and never re-dispatches a payout.
        """
        binding = self.store.operation_binding(operation_key)
        if binding["state"] not in OBSERVABLE_OPERATION_STATES:
            raise ValueError("OBSERVABLE_OPERATION_REQUIRED")
        payout_batch_id = binding["payout_batch_id"]
        if not isinstance(payout_batch_id, str) or not payout_batch_id:
            raise ValueError("PAYOUT_BATCH_BINDING_REQUIRED")
        intent = self.store.load_intent(operation_key)

        batch = self.adapter.get_batch(payout_batch_id, access_token=access_token)
        items, failure = self._batch_items(operation_key, batch, payout_batch_id)
        if failure is not None:
            return self._record_conflict(operation_key, failure)
        matching = self._matching_items(items, intent)
        if len(matching) != 1:
            return self._record_conflict(operation_key, "OBSERVED_BATCH_ITEM_MISMATCH")
        payout_item_id = matching[0].get("payout_item_id")
        if not isinstance(payout_item_id, str) or not payout_item_id:
            return self._record_conflict(operation_key, "OBSERVED_ITEM_ID_MISSING")

        item = self.adapter.get_item(payout_item_id, access_token=access_token)
        if item.get("payout_item_id") != payout_item_id:
            return self._record_conflict(operation_key, "OBSERVED_ITEM_DETAIL_MISMATCH")
        payout = item.get("payout_item")
        if not isinstance(payout, dict) or payout.get("sender_item_id") != self._sender_item_id(intent):
            return self._record_conflict(operation_key, "OBSERVED_SENDER_ITEM_ID_MISMATCH")

        # A different payout_item_id for the same claim identity is retained as
        # evidence; the store records it as PAYOUT_ITEM_ID_MISMATCH and keeps the
        # operation's original item binding.
        return self._persist_item_observation(operation_key, payout_batch_id, payout_item_id, item)

    def _persist_item_observation(
        self,
        operation_key: str,
        payout_batch_id: str,
        payout_item_id: str,
        item: dict[str, object],
    ) -> RecoveryDisposition:
        try:
            status = ProviderItemStatus(item["transaction_status"])
            payout = item["payout_item"]
            if not isinstance(payout, dict):
                raise ValueError("RECOVERED_ITEM_DETAIL_INVALID")
            amount = self._money(payout["amount"])
            recipient = payout["receiver"]
            if not isinstance(recipient, str):
                raise ValueError("RECOVERED_RECIPIENT_INVALID")
        except (KeyError, TypeError, ValueError):
            return self._record_conflict(operation_key, "RECOVERED_ITEM_DETAIL_INVALID")

        raw = json.dumps(item, sort_keys=True, separators=(",", ":")).encode("utf-8")
        observation = ProviderObservation(
            observation_id=f"observation-{uuid4().hex}",
            attempt_id=operation_key,
            provider="PAYPAL_SANDBOX",
            payout_batch_id=payout_batch_id,
            payout_item_id=payout_item_id,
            status=status,
            recipient=recipient,
            amount=amount,
            observed_at=datetime.now(timezone.utc),
            raw_sha256=sha256(raw).hexdigest(),
        )
        intent = self.store.load_intent(operation_key)
        self.store.persist_reconciliation(
            operation_key, observation, reconcile_payment(intent, observation)
        )
        return self._disposition(self.store.latest_financial_outcome(operation_key)["outcome"])

    @staticmethod
    def _disposition(outcome: object) -> RecoveryDisposition:
        if outcome == "matched_completed":
            return RecoveryDisposition.RECONCILED
        if outcome == "pending":
            return RecoveryDisposition.PENDING
        if outcome == "failed":
            # Completed evidence was superseded by a terminal reversal/refund/return.
            return RecoveryDisposition.SUPERSEDED
        return RecoveryDisposition.CONFLICT

    def _record_conflict(self, operation_key: str, detail: str) -> RecoveryDisposition:
        if self.store.operation_state(operation_key) == "UNCERTAIN":
            self.store.mark_recovery_conflict(operation_key, detail)
        else:
            self.store.record_observation_conflict(operation_key, detail)
        return RecoveryDisposition.CONFLICT

    def _batch_items(
        self,
        operation_key: str,
        batch: object,
        payout_batch_id: str,
    ) -> tuple[list[object], str | None]:
        try:
            header = batch["batch_header"]  # type: ignore[index]
            observed_batch_id = header["payout_batch_id"]
            sender_batch_id = header["sender_batch_header"]["sender_batch_id"]
            items = batch["items"]  # type: ignore[index]
        except (KeyError, TypeError):
            return [], "RECOVERED_BATCH_SHAPE_INVALID"
        if observed_batch_id != payout_batch_id or sender_batch_id != self._sender_batch_id_for_key(
            operation_key
        ):
            return [], "RECOVERED_BATCH_IDENTITY_MISMATCH"
        if not isinstance(items, list):
            return [], "RECOVERED_BATCH_SHAPE_INVALID"
        return items, None

    @staticmethod
    def _matching_items(items: list[object], intent: PaymentIntent) -> list[dict[str, object]]:
        expected = PayPalRecovery._sender_item_id(intent)
        return [
            item
            for item in items
            if isinstance(item, dict)
            and isinstance(item.get("payout_item"), dict)
            and item["payout_item"].get("sender_item_id") == expected
        ]

    @staticmethod
    def _sender_item_id(intent: PaymentIntent) -> str:
        return f"{intent.claim_id}-r{intent.claim_revision}"

    @staticmethod
    def _sender_batch_id_for_key(operation_key: str) -> str:
        digest = operation_key.removeprefix("missionpay:")
        return f"mp-{digest[:27]}"

    @classmethod
    def _item_matches_intent(cls, item: dict[str, object], intent: PaymentIntent) -> bool:
        try:
            payout = item["payout_item"]
            if not isinstance(payout, dict):
                return False
            return (
                payout["sender_item_id"] == cls._sender_item_id(intent)
                and payout["receiver"] == intent.recipient
                and cls._money(payout["amount"]) == intent.amount
            )
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _money(value: object) -> Money:
        if not isinstance(value, dict) or value.get("currency") != "USD":
            raise ValueError("RECOVERED_CURRENCY_INVALID")
        try:
            decimal = Decimal(value["value"])
        except (InvalidOperation, KeyError, TypeError) as exc:
            raise ValueError("RECOVERED_AMOUNT_INVALID") from exc
        cents = decimal * 100
        if cents != cents.to_integral_value():
            raise ValueError("RECOVERED_AMOUNT_PRECISION_INVALID")
        return Money("USD", int(cents))
