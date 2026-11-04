"""The payment service the web layer talks to.

Responsibilities are deliberately narrow and ordered by trust:

* the web layer asks this object to execute or observe; it never sees a client
  secret, a bearer token, or a transport;
* in ``mock`` mode every call refuses, so the product cannot silently move money
  because an environment variable was left set;
* in ``paypal_sandbox`` mode the bearer token is acquired immediately before the
  dispatch, from the single component allowed to hold the secret.

Mock mode is not a simulation of success. It returns a refusal, and the UI renders
that refusal. A demo that wanted to *show* a payout must therefore be run against
the sandbox, which is the honest way to show one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config import ProviderMode, RuntimeConfig
from .domain import PaymentIntent
from .executor import DispatchOutcome, DispatchRefused, PayoutExecutor
from .oauth import PayPalClientCredentials
from .recovery import PayPalRecovery, RecoveryDisposition
from .store import MissionPayStore


class PaymentServiceRefused(RuntimeError):
    """The configured mode does not permit this operation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ServiceReport:
    provider_mode: str
    is_genuine: bool
    label: str


class PaymentService:
    def __init__(
        self,
        store: MissionPayStore,
        config: RuntimeConfig,
        *,
        adapter: object | None = None,
        executor: PayoutExecutor | None = None,
        recovery: PayPalRecovery | None = None,
        token_provider: Callable[[], str] | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self._adapter = adapter
        self._executor = executor
        self._recovery = recovery
        self._token_provider = token_provider
        self._credentials: PayPalClientCredentials | None = None

    # -- introspection ---------------------------------------------------

    def report(self) -> ServiceReport:
        return ServiceReport(
            provider_mode=self.config.provider_mode.value,
            is_genuine=self.config.is_genuine,
            label=self.config.mode_label,
        )

    def can_execute(self) -> bool:
        return self.config.provider_mode is ProviderMode.PAYPAL_SANDBOX

    # -- lazy construction (only reached in genuine mode) ----------------

    def _require_genuine(self) -> None:
        if self.config.provider_mode is not ProviderMode.PAYPAL_SANDBOX:
            raise PaymentServiceRefused("MOCK_MODE_REFUSES_PROVIDER_OPERATIONS")

    def _token(self) -> str:
        if self._token_provider is not None:
            return self._token_provider()
        # One client instance for the service's lifetime, so the bounded token cache
        # actually applies. Constructing a client per call minted a fresh token for
        # every dispatch and readback.
        if self._credentials is None:
            self._credentials = PayPalClientCredentials(
                client_id=self.config.paypal_client_id or "",
                client_secret=self.config.paypal_client_secret or "",
                base_url=self.config.paypal_base_url,
            )
        return self._credentials.access_token()

    def token_report(self) -> dict[str, object]:
        """Non-secret token summary. Only meaningful in genuine mode."""
        self._require_genuine()
        self._token()
        assert self._credentials is not None
        cached = self._credentials.cached_token()
        return cached.describe() if cached is not None else {"authentication": "unknown"}

    def _adapter_or_default(self):
        """One adapter instance for dispatch and readback, so one origin policy applies."""
        if self._adapter is None:
            from .paypal import PayPalSandboxAdapter

            self._adapter = PayPalSandboxAdapter(base_url=self.config.paypal_base_url)
        return self._adapter

    def _get_executor(self) -> PayoutExecutor:
        if self._executor is None:
            self._executor = PayoutExecutor(
                self.store,
                self._adapter_or_default(),
                max_payout_cents=self.config.max_payout_cents,
                expected_recipient=self.config.sandbox_recipient,
            )
        return self._executor

    def _get_recovery(self) -> PayPalRecovery:
        if self._recovery is None:
            self._recovery = PayPalRecovery(self.store, self._adapter_or_default())
        return self._recovery

    # -- operations ------------------------------------------------------

    def execute(self, operation_key: str, *, owner_id: str) -> DispatchOutcome:
        self._require_genuine()
        return self._get_executor().execute(
            operation_key,
            token_provider=self._token,
            owner_id=owner_id,
        )

    def preflight(self, operation_key: str) -> PaymentIntent:
        self._require_genuine()
        return self._get_executor().preflight(operation_key)

    def observe(self, operation_key: str) -> RecoveryDisposition:
        self._require_genuine()
        return self._get_recovery().observe_known_payout(
            operation_key, access_token=self._token()
        )

    def attach_batch(self, operation_key: str, payout_batch_id: str) -> RecoveryDisposition:
        self._require_genuine()
        return self._get_recovery().recover_with_batch_id(
            operation_key, payout_batch_id, access_token=self._token()
        )

    def recover_interrupted(self) -> int:
        """Convert leftover SENDING rows to UNCERTAIN. Runs at startup, any mode."""
        return self.store.recover_interrupted_dispatches()


__all__ = [
    "DispatchRefused",
    "PaymentService",
    "PaymentServiceRefused",
    "ServiceReport",
]
