"""Uvicorn entrypoint for MissionPay.

Configuration is resolved once at import and fails closed here, before the
application object exists, so a misconfigured genuine run cannot start serving
pages that imply a capability it does not have.

Startup also performs the restart-recovery step: any operation left in ``SENDING``
by a process that died mid-dispatch is converted to ``UNCERTAIN``. That is the only
safe reading of a durable dispatch claim with no recorded provider response, and it
happens in every mode because it is a store transition, not a provider call.
"""

from __future__ import annotations

from pathlib import Path

from .bootstrap import bootstrap_demo_store
from .config import from_environment
from .service import PaymentService
from .store import MissionPayStore
from .web import create_app

config = from_environment()
_database = Path(config.database_path)

if config.demo_bootstrap:
    store = bootstrap_demo_store(_database)
else:
    store = MissionPayStore(_database)
    store.initialize()

recovered_dispatches = PaymentService(store, config).recover_interrupted()

app = create_app(store, config=config)
