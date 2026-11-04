"""Idempotent local-only fixture bootstrap for the focused demo."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import sqlite3

from .auth import Role
from .domain import FundPolicy, Money
from .store import MissionPayStore


def bootstrap_demo_store(path: str | Path) -> MissionPayStore:
    store = MissionPayStore(path)
    store.initialize()
    for user in (
        ("demo-volunteer", "volunteer@missionpay.local", Role.VOLUNTEER, "volunteer-demo-passphrase"),
        ("demo-reviewer", "reviewer@missionpay.local", Role.REVIEWER, "reviewer-demo-passphrase"),
    ):
        try:
            store.register_user(*user)
        except sqlite3.IntegrityError:
            pass
    for policy in (
        FundPolicy(
            fund_id="GARDEN_FUND",
            version="2026.1",
            allowed_activity_codes=frozenset({"COMMUNITY_GARDEN"}),
            allowed_expense_categories=frozenset({"SEEDS"}),
            starts_on=date(2026, 1, 1),
            ends_on=date(2026, 12, 31),
            available=Money("USD", 50_000),
            receipt_required=True,
        ),
        FundPolicy(
            fund_id="RIVER_CLEANUP_FUND",
            version="2026.1",
            allowed_activity_codes=frozenset({"RIVER_CLEANUP"}),
            allowed_expense_categories=frozenset({"PROTECTIVE_SUPPLIES"}),
            starts_on=date(2026, 1, 1),
            ends_on=date(2026, 12, 31),
            available=Money("USD", 50_000),
            receipt_required=True,
        ),
    ):
        try:
            store.create_fund(policy)
        except sqlite3.IntegrityError:
            pass
    return store
