"""SQLite persistence and authenticated authority boundaries for MissionPay."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import hmac
from pathlib import Path
import secrets
import sqlite3
from typing import Iterator
from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from .ai import CITATION_FAILURES, DIAGNOSTIC_CODES, FACT_NAMES
from .auth import (
    AuthenticatedSession,
    AuthenticationFailed,
    AuthorizationDenied,
    Role,
)
from .domain import (
    Activity,
    Claim,
    EvidenceIdentity,
    FundBalance,
    FundPolicy,
    HumanAuthorization,
    Money,
    PaymentIntent,
    PolicyDecision,
    PolicyOutcome,
    ProviderObservation,
    Reconciliation,
    ReconciliationOutcome,
)
from .payment import create_payment_intent as bind_payment_intent


class StaleAuthorization(ValueError):
    pass


class PersistenceConflict(ValueError):
    pass


class InsufficientFunds(ValueError):
    pass


class DispatchBlocked(ValueError):
    pass


#: Terminal states a recorded hosted attempt may move to. ``RECORDED`` is the only
#: non-terminal state, and it is written before the request leaves.
_HOSTED_ATTEMPT_TERMINAL = frozenset({"SUCCEEDED", "FAILED", "TIMEOUT", "UNCERTAIN"})

#: A hosted attempt is pinned by its authorization reference. The columns that define
#: what was approved cannot be rewritten afterwards; only the outcome may be filled in.
_HOSTED_ATTEMPTS_DDL = """CREATE TABLE IF NOT EXISTS hosted_attempts (
    authorization_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    claim_revision INTEGER NOT NULL,
    evidence_sha256 TEXT NOT NULL,
    expectation_sha256 TEXT NOT NULL,
    model_id TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    max_output_tokens INTEGER NOT NULL,
    max_total_tokens INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('RECORDED','SUCCEEDED','FAILED','TIMEOUT','UNCERTAIN')
    ),
    code TEXT,
    usage_json TEXT,
    diagnostic_json TEXT,
    recorded_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TRIGGER IF NOT EXISTS hosted_attempts_no_delete
BEFORE DELETE ON hosted_attempts
BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_HOSTED_ATTEMPT'); END;
CREATE TRIGGER IF NOT EXISTS hosted_attempts_pinned
BEFORE UPDATE ON hosted_attempts
WHEN OLD.authorization_id IS NOT NEW.authorization_id
  OR OLD.claim_id IS NOT NEW.claim_id
  OR OLD.claim_revision IS NOT NEW.claim_revision
  OR OLD.evidence_sha256 IS NOT NEW.evidence_sha256
  OR OLD.expectation_sha256 IS NOT NEW.expectation_sha256
  OR OLD.model_id IS NOT NEW.model_id
  OR OLD.config_sha256 IS NOT NEW.config_sha256
  OR OLD.max_output_tokens IS NOT NEW.max_output_tokens
  OR OLD.max_total_tokens IS NOT NEW.max_total_tokens
  OR OLD.recorded_at IS NOT NEW.recorded_at
BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_HOSTED_ATTEMPT_PINNING'); END;
"""

_PROVIDER_OBSERVATIONS_DDL = """CREATE TABLE IF NOT EXISTS provider_observations (
    observation_id TEXT PRIMARY KEY,
    operation_key TEXT NOT NULL REFERENCES payment_operations(operation_key),
    payout_batch_id TEXT NOT NULL,
    payout_item_id TEXT NOT NULL,
    status TEXT NOT NULL,
    recipient TEXT NOT NULL,
    currency TEXT NOT NULL CHECK (currency = 'USD'),
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
    observed_at TEXT NOT NULL,
    raw_sha256 TEXT NOT NULL,
    UNIQUE (operation_key, payout_item_id, raw_sha256)
)"""

_RECONCILIATIONS_DDL = """CREATE TABLE IF NOT EXISTS reconciliations (
    reconciliation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_key TEXT NOT NULL REFERENCES payment_operations(operation_key),
    observation_id TEXT NOT NULL UNIQUE REFERENCES provider_observations(observation_id),
    outcome TEXT NOT NULL,
    reason_codes_json TEXT NOT NULL,
    reconciled_at TEXT NOT NULL
)"""

_LIFECYCLE_INDEX_DDL = (
    """CREATE INDEX IF NOT EXISTS provider_observations_by_item
       ON provider_observations(payout_item_id, observed_at)""",
    """CREATE INDEX IF NOT EXISTS reconciliations_by_operation
       ON reconciliations(operation_key, reconciliation_id)""",
)

_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    role TEXT NOT NULL CHECK (role IN ('volunteer','reviewer')),
    password_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    csrf_hash TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    role TEXT NOT NULL CHECK (role IN ('volunteer','reviewer')),
    expires_at TEXT NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0 CHECK (revoked IN (0,1))
);
CREATE TABLE IF NOT EXISTS evidence (
    sha256 TEXT PRIMARY KEY,
    media_type TEXT NOT NULL,
    original_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fund_policies (
    fund_id TEXT NOT NULL,
    version TEXT NOT NULL,
    allowed_activities_json TEXT NOT NULL,
    allowed_categories_json TEXT NOT NULL,
    starts_on TEXT NOT NULL,
    ends_on TEXT NOT NULL,
    currency TEXT NOT NULL CHECK (currency = 'USD'),
    available_cents INTEGER NOT NULL CHECK (available_cents > 0),
    reserved_cents INTEGER NOT NULL DEFAULT 0 CHECK (reserved_cents >= 0),
    receipt_required INTEGER NOT NULL CHECK (receipt_required IN (0,1)),
    PRIMARY KEY (fund_id, version),
    CHECK (reserved_cents <= available_cents)
);
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    volunteer_id TEXT NOT NULL REFERENCES users(user_id),
    activity_id TEXT NOT NULL,
    activity_code TEXT NOT NULL,
    occurred_on TEXT NOT NULL,
    evidence_sha256 TEXT NOT NULL REFERENCES evidence(sha256),
    currency TEXT NOT NULL CHECK (currency = 'USD'),
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
    recipient TEXT NOT NULL,
    expense_category TEXT,
    PRIMARY KEY (claim_id, revision)
);
CREATE TABLE IF NOT EXISTS policy_decisions (
    claim_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    fund_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('eligible','ineligible','needs_review')),
    reason_codes_json TEXT NOT NULL,
    PRIMARY KEY (claim_id, revision, fund_id, policy_version),
    FOREIGN KEY (claim_id, revision) REFERENCES claims(claim_id, revision),
    FOREIGN KEY (fund_id, policy_version) REFERENCES fund_policies(fund_id, version)
);
CREATE TABLE IF NOT EXISTS authorizations (
    authorization_id TEXT PRIMARY KEY,
    reviewer_id TEXT NOT NULL REFERENCES users(user_id),
    claim_id TEXT NOT NULL,
    claim_revision INTEGER NOT NULL,
    evidence_sha256 TEXT NOT NULL,
    fund_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    recipient TEXT NOT NULL,
    currency TEXT NOT NULL CHECK (currency = 'USD'),
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
    created_at TEXT NOT NULL,
    UNIQUE (claim_id, claim_revision, fund_id, policy_version),
    FOREIGN KEY (claim_id, claim_revision) REFERENCES claims(claim_id, revision),
    FOREIGN KEY (fund_id, policy_version) REFERENCES fund_policies(fund_id, version)
);
CREATE TABLE IF NOT EXISTS interpretations (
    interpretation_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    claim_revision INTEGER NOT NULL,
    evidence_sha256 TEXT NOT NULL,
    ai_mode TEXT NOT NULL,
    model_id TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1 CHECK (attempt >= 1),
    status TEXT NOT NULL CHECK (status IN ('ACCEPTED','REFUSED')),
    error_code TEXT,
    output_sha256 TEXT NOT NULL,
    facts_json TEXT NOT NULL DEFAULT '[]',
    missing_json TEXT NOT NULL DEFAULT '[]',
    -- Which interpretation contract this answer was judged under. NULL means the record
    -- predates the column, so it was never checked for a unique quote; the page must say
    -- that rather than imply today's rule applied.
    contract_version TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (claim_id, claim_revision, evidence_sha256, ai_mode, model_id, config_sha256, attempt),
    FOREIGN KEY (claim_id, claim_revision) REFERENCES claims(claim_id, revision)
);
CREATE TABLE IF NOT EXISTS payment_intents (
    operation_key TEXT PRIMARY KEY,
    authorization_id TEXT NOT NULL UNIQUE REFERENCES authorizations(authorization_id),
    claim_id TEXT NOT NULL,
    claim_revision INTEGER NOT NULL,
    evidence_sha256 TEXT NOT NULL UNIQUE,
    fund_id TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    recipient TEXT NOT NULL,
    currency TEXT NOT NULL CHECK (currency = 'USD'),
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
    created_at TEXT NOT NULL,
    UNIQUE (claim_id, claim_revision),
    FOREIGN KEY (claim_id, claim_revision) REFERENCES claims(claim_id, revision)
);
CREATE TABLE IF NOT EXISTS payment_operations (
    operation_key TEXT PRIMARY KEY REFERENCES payment_intents(operation_key),
    request_id TEXT NOT NULL UNIQUE,
    sender_batch_id TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (state IN (
        'PREPARED','SENDING','UNCERTAIN','ACKNOWLEDGED','VERIFIED','CONFLICT'
    )),
    owner_id TEXT,
    payout_batch_id TEXT UNIQUE,
    payout_item_id TEXT UNIQUE,
    detail TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
""" + _PROVIDER_OBSERVATIONS_DDL + ";" + _RECONCILIATIONS_DDL + ";" + _HOSTED_ATTEMPTS_DDL + ";" + """
CREATE TABLE IF NOT EXISTS audit_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    aggregate_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS evidence_no_update
BEFORE UPDATE ON evidence BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_EVIDENCE'); END;
CREATE TRIGGER IF NOT EXISTS evidence_no_delete
BEFORE DELETE ON evidence BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_EVIDENCE'); END;
CREATE TRIGGER IF NOT EXISTS claims_no_update
BEFORE UPDATE ON claims BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_CLAIM_REVISION'); END;
CREATE TRIGGER IF NOT EXISTS claims_no_delete
BEFORE DELETE ON claims BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_CLAIM_REVISION'); END;
CREATE TRIGGER IF NOT EXISTS decisions_no_update
BEFORE UPDATE ON policy_decisions BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_POLICY_DECISION'); END;
CREATE TRIGGER IF NOT EXISTS decisions_no_delete
BEFORE DELETE ON policy_decisions BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_POLICY_DECISION'); END;
CREATE TRIGGER IF NOT EXISTS authorizations_no_update
BEFORE UPDATE ON authorizations BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_AUTHORIZATION'); END;
CREATE TRIGGER IF NOT EXISTS authorizations_no_delete
BEFORE DELETE ON authorizations BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_AUTHORIZATION'); END;
CREATE TRIGGER IF NOT EXISTS intents_no_update
BEFORE UPDATE ON payment_intents BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_PAYMENT_INTENT'); END;
CREATE TRIGGER IF NOT EXISTS intents_no_delete
BEFORE DELETE ON payment_intents BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_PAYMENT_INTENT'); END;
CREATE TRIGGER IF NOT EXISTS observations_no_update
BEFORE UPDATE ON provider_observations BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_PROVIDER_OBSERVATION'); END;
CREATE TRIGGER IF NOT EXISTS observations_no_delete
BEFORE DELETE ON provider_observations BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_PROVIDER_OBSERVATION'); END;
CREATE TRIGGER IF NOT EXISTS reconciliations_no_update
BEFORE UPDATE ON reconciliations BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_RECONCILIATION'); END;
CREATE TRIGGER IF NOT EXISTS reconciliations_no_delete
BEFORE DELETE ON reconciliations BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_RECONCILIATION'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_update
BEFORE UPDATE ON audit_events BEGIN SELECT RAISE(ABORT, 'APPEND_ONLY_AUDIT'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete
BEFORE DELETE ON audit_events BEGIN SELECT RAISE(ABORT, 'APPEND_ONLY_AUDIT'); END;
"""


# An observation may be appended once the payout batch is authoritatively
# acknowledged, and on every later derived state of the same operation. PREPARED,
# SENDING and UNCERTAIN remain closed because no authoritative batch binding exists.
OBSERVABLE_OPERATION_STATES = frozenset({"ACKNOWLEDGED", "VERIFIED", "CONFLICT"})

# Outcomes that only exist for databases written by the pre-MP-LIFECYCLE-006 schema.
_LEGACY_TABLE_MARKERS = (
    ("provider_observations", "payout_item_id TEXT NOT NULL UNIQUE"),
    ("reconciliations", "operation_key TEXT PRIMARY KEY"),
)

# Operation phase state is derived from reconciled observation history. VERIFIED
# means "provider truth independently established", not "money delivered"; the
# financial outcome (paid / pending / failed / refunded / reversed / returned) is
# exposed separately by latest_financial_outcome().
_OPERATION_STATE_BY_OUTCOME = {
    "matched_completed": "VERIFIED",
    "failed": "VERIFIED",
    "pending": "ACKNOWLEDGED",
    "mismatch": "CONFLICT",
    "uncertain": "CONFLICT",
}


@dataclass(frozen=True, slots=True)
class _ItemObservationState:
    """Latest authoritative observation of one provider payout item."""

    payout_item_id: str
    observation_id: str
    reconciliation_id: int
    status: str
    outcome: str
    reason_codes: tuple[str, ...]
    observed_at: datetime
    raw_sha256: str
    recipient: str
    amount_cents: int


def _constant_time_equal(left: str | None, right: str | None) -> bool:
    """Compare two identifiers without leaking a prefix match through timing."""
    if left is None or right is None:
        return False
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


#: The one ordering rule for "which accepted interpretation is current".
#:
#: ``attempt`` is per configuration, so it cannot rank two different configurations
#: (A's attempt 2 is not newer than B's attempt 1). Recency is therefore the primary
#: key; attempt and id only break ties inside the same instant, which happens when two
#: rows share a timestamp. Every call site uses this fragment through
#: :func:`_latest_accepted`, so the page a reviewer sees and the authority that gets
#: bound can never disagree.
_ACCEPTED_ORDER = "ORDER BY created_at DESC, attempt DESC, interpretation_id DESC"

#: Within one configuration, attempt is the meaningful rank: a later attempt supersedes
#: an earlier one even at the same timestamp.
_ATTEMPT_ORDER = "ORDER BY attempt DESC, created_at DESC"


#: The only keys a citation diagnostic may carry. Exact values from declared sets, not
#: identifier-shaped patterns: ``^[A-Za-z_]+$`` also matches ``SyntheticSecret`` and
#: ``sk_live_...``, so "looks like a name" is not the same guarantee as "is one of these".
_DIAGNOSTIC_CODE_VALUES = DIAGNOSTIC_CODES
_DIAGNOSTIC_CATEGORY_VALUES = CITATION_FAILURES
_DIAGNOSTIC_FACT_VALUES = FACT_NAMES + ("unrecognised",)
_DIAGNOSTIC_INT_KEYS = (
    "start",
    "end",
    "quote_length",
    "evidence_length",
    "occurrences",
    # The catalogue number the answer chose. A number, so it says which excerpt was meant
    # without saying what it contained.
    "excerpt_id",
)
#: Flags are recorded only when true; the absence of a flag is the ordinary case.
_DIAGNOSTIC_FLAG_KEYS = ("unicode_duplicate",)


def clean_diagnostic(diagnostic: object) -> dict[str, object] | None:
    """Reduce a diagnostic to declared identifiers and bounded integers.

    The one sanitiser: the journal, the audit event, the CLI and the outcome file all use
    its output, so no surface can show a field another surface dropped. Anything that is
    not an exact member of a declared set — or an integer in range — is discarded, and a
    diagnostic without a valid code, category and fact name is discarded whole rather
    than half-recorded.
    """
    if not isinstance(diagnostic, dict):
        return None
    code = diagnostic.get("code")
    category = diagnostic.get("category")
    fact_name = diagnostic.get("fact_name")
    if code not in _DIAGNOSTIC_CODE_VALUES:
        return None
    if category not in _DIAGNOSTIC_CATEGORY_VALUES:
        return None
    if fact_name not in _DIAGNOSTIC_FACT_VALUES:
        return None
    cleaned: dict[str, object] = {
        "code": code,
        "fact_name": fact_name,
        "category": category,
    }
    for key in _DIAGNOSTIC_INT_KEYS:
        value = diagnostic.get(key)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and -10**9 <= value <= 10**9
        ):
            cleaned[key] = value
    for key in _DIAGNOSTIC_FLAG_KEYS:
        if diagnostic.get(key) is True:
            cleaned[key] = True
    return cleaned


def _hosted_attempt_row(row: sqlite3.Row) -> dict[str, object]:
    """Shape a hosted attempt for reporting. Usage stays structured, never free text."""
    usage = json.loads(row["usage_json"]) if row["usage_json"] else None
    diagnostic = json.loads(row["diagnostic_json"]) if row["diagnostic_json"] else None
    return {
        "authorization_id": row["authorization_id"],
        "claim_id": row["claim_id"],
        "claim_revision": row["claim_revision"],
        "evidence_sha256": row["evidence_sha256"],
        "expectation_sha256": row["expectation_sha256"],
        "model_id": row["model_id"],
        "config_sha256": row["config_sha256"],
        "max_output_tokens": row["max_output_tokens"],
        "max_total_tokens": row["max_total_tokens"],
        "state": row["state"],
        "code": row["code"],
        "usage": usage,
        "diagnostic": diagnostic,
        "recorded_at": row["recorded_at"],
        "finished_at": row["finished_at"],
    }


def _latest_accepted(
    conn: sqlite3.Connection,
    claim_id: str,
    revision: int,
    evidence_sha256: str | None = None,
) -> sqlite3.Row | None:
    query = """SELECT * FROM interpretations
               WHERE claim_id=? AND claim_revision=? AND status='ACCEPTED'"""
    params: list[object] = [claim_id, revision]
    if evidence_sha256 is not None:
        query += " AND evidence_sha256=?"
        params.append(evidence_sha256)
    query += f" {_ACCEPTED_ORDER} LIMIT 1"
    return conn.execute(query, tuple(params)).fetchone()


class MissionPayStore:
    def __init__(self, path: str | Path, *, password_hasher: PasswordHasher | None = None):
        self.path = str(path)
        self.password_hasher = password_hasher or PasswordHasher()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 10000")
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = FULL")
            conn.executescript(_SCHEMA)
            self._ensure_hosted_attempt_columns(conn)
            self._migrate_lifecycle_schema(conn)
            self._ensure_interpretation_binding(conn)
            self._ensure_interpretation_attempts(conn)
            self._ensure_interpretation_contract(conn)
            for statement in _LIFECYCLE_INDEX_DDL:
                conn.execute(statement)

    @staticmethod
    def _ensure_interpretation_contract(conn: sqlite3.Connection) -> None:
        """Add the contract-version column to a database written before it existed.

        It runs after the attempts rebuild, because that rebuild recreates the table from
        an explicit column list and would drop anything added before it. Existing rows keep
        NULL, which is the point: the value records what a record was judged under, and an
        old record was not judged under today's rule.
        """
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(interpretations)").fetchall()
        }
        if "contract_version" not in columns:
            conn.execute("ALTER TABLE interpretations ADD COLUMN contract_version TEXT")

    @staticmethod
    def _ensure_hosted_attempt_columns(conn: sqlite3.Connection) -> None:
        """Add columns a journal from an earlier candidate would lack.

        ``CREATE TABLE IF NOT EXISTS`` cannot add a column, so a database written before
        this correction would silently lack the binding. A NULL default would make the
        pinning trigger meaningless, so the expectation column is added with an empty
        string and any pre-existing row is refused for qualification by the caller rather
        than trusted. Diagnostics are nullable: an attempt that never failed has none.
        """
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(hosted_attempts)").fetchall()
        }
        if "expectation_sha256" not in columns:
            conn.execute(
                "ALTER TABLE hosted_attempts ADD COLUMN expectation_sha256 TEXT NOT NULL DEFAULT ''"
            )
        if "diagnostic_json" not in columns:
            conn.execute("ALTER TABLE hosted_attempts ADD COLUMN diagnostic_json TEXT")

    @staticmethod
    def _ensure_interpretation_attempts(conn: sqlite3.Connection) -> None:
        """Rebuild the interpretations table so refused attempts can be retried.

        The first version keyed an interpretation on (claim, revision, evidence, mode,
        model, config) with no notion of an attempt, so a REFUSED result permanently
        occupied the key and a later successful retry returned the stale refusal. The
        unique constraint cannot be altered in place, so the table is rebuilt with an
        ``attempt`` column; existing rows are preserved as attempt 1 in creation order.

        The rebuild runs inside one explicit transaction. ``executescript`` cannot be
        used here: it issues its own COMMIT and then runs the statements in autocommit
        mode, so a crash between ``DROP TABLE interpretations`` and the rename leaves
        the copy orphaned, the next start creates an empty table from the schema and
        every interpretation and approval binding is silently lost. On this connection
        ``isolation_level`` is None, so SQLite's own transactional DDL applies and the
        drop, the copy and the rename either all happen or none of them do.
        """
        MissionPayStore._recover_interrupted_attempt_migration(conn)

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(interpretations)")}
        if not columns or "attempt" in columns:
            return

        conn.execute("BEGIN IMMEDIATE")
        try:
            MissionPayStore._create_migrated_interpretations(conn)
            MissionPayStore._fill_migrated_interpretations(conn)
            MissionPayStore._drop_unmigrated_interpretations(conn)
            MissionPayStore._rename_migrated_interpretations(conn)
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    # -- migration steps, separate so tests can inject a failure at each boundary --

    @staticmethod
    def _create_migrated_interpretations(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE interpretations_migrated (
                interpretation_id TEXT PRIMARY KEY,
                claim_id TEXT NOT NULL,
                claim_revision INTEGER NOT NULL,
                evidence_sha256 TEXT NOT NULL,
                ai_mode TEXT NOT NULL,
                model_id TEXT NOT NULL,
                config_sha256 TEXT NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 1 CHECK (attempt >= 1),
                status TEXT NOT NULL CHECK (status IN ('ACCEPTED','REFUSED')),
                error_code TEXT,
                output_sha256 TEXT NOT NULL,
                facts_json TEXT NOT NULL DEFAULT '[]',
                missing_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                UNIQUE (claim_id, claim_revision, evidence_sha256, ai_mode, model_id,
                        config_sha256, attempt),
                FOREIGN KEY (claim_id, claim_revision) REFERENCES claims(claim_id, revision)
            )
            """
        )

    @staticmethod
    def _fill_migrated_interpretations(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO interpretations_migrated(
                interpretation_id,claim_id,claim_revision,evidence_sha256,ai_mode,model_id,
                config_sha256,attempt,status,error_code,output_sha256,facts_json,missing_json,
                created_at
            )
            SELECT interpretation_id,claim_id,claim_revision,evidence_sha256,ai_mode,model_id,
                   config_sha256,
                   ROW_NUMBER() OVER (
                       PARTITION BY claim_id,claim_revision,evidence_sha256,ai_mode,model_id,
                                    config_sha256
                       ORDER BY created_at, interpretation_id
                   ),
                   status,error_code,output_sha256,facts_json,missing_json,created_at
            FROM interpretations
            """
        )

    @staticmethod
    def _drop_unmigrated_interpretations(conn: sqlite3.Connection) -> None:
        conn.execute("DROP TABLE interpretations")

    @staticmethod
    def _rename_migrated_interpretations(conn: sqlite3.Connection) -> None:
        conn.execute("ALTER TABLE interpretations_migrated RENAME TO interpretations")

    @staticmethod
    def _recover_interrupted_attempt_migration(conn: sqlite3.Connection) -> None:
        """Repair a database left behind by the non-atomic version of this migration.

        A build before this fix could die after the copy and the DROP but before the
        rename. The copy then holds every row, the schema creates an empty
        ``interpretations`` on the next start, and the attempt-column check sees an
        already-migrated table and returns.

        Recovery is only automatic where it is provably lossless:

        * live table missing  -> the copy is promoted by rename;
        * live table empty    -> the copy is promoted inside one transaction;
        * both hold rows      -> the copy is discarded **only after proving** that every
          one of its records is already present in the live table with identical
          contents. Any other outcome keeps both tables and refuses to start.

        The last case is a bounded refusal rather than a merge: deciding how to combine
        two disagreeing histories is a human judgement about financial records, and
        silently dropping the orphan is exactly the data loss this function exists to
        prevent.
        """

        def table_exists(name: str) -> bool:
            return conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone() is not None

        def row_count(name: str) -> int:
            return conn.execute(f"SELECT COUNT(*) AS n FROM {name}").fetchone()["n"]

        def columns(name: str) -> list[str]:
            return [row["name"] for row in conn.execute(f"PRAGMA table_info({name})")]

        if not table_exists("interpretations_migrated"):
            return

        if not table_exists("interpretations"):
            conn.execute("ALTER TABLE interpretations_migrated RENAME TO interpretations")
            return

        live = row_count("interpretations")
        orphaned = row_count("interpretations_migrated")

        if orphaned == 0:
            # Nothing can be lost: either the copy is empty or it never held anything.
            conn.execute("DROP TABLE interpretations_migrated")
            return

        if live == 0:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DROP TABLE interpretations")
                conn.execute("ALTER TABLE interpretations_migrated RENAME TO interpretations")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            return

        # Both tables hold rows. Only discard the copy if it is provably a duplicate.
        MissionPayStore._assert_orphan_copy_is_duplicate(
            conn, columns("interpretations"), columns("interpretations_migrated")
        )
        conn.execute("DROP TABLE interpretations_migrated")

    @staticmethod
    def _assert_orphan_copy_is_duplicate(
        conn: sqlite3.Connection, live_columns: list[str], orphan_columns: list[str]
    ) -> None:
        """Prove every orphan record also exists, identically, in the live table.

        Comparison covers the columns the two tables share. A column that exists only
        in the copy cannot be compared, so it is treated as unprovable and refuses
        rather than guessing. Raises :class:`PersistenceConflict` on any mismatch; the
        caller has performed reads only, so both tables are left untouched.
        """
        uncomparable = sorted(set(orphan_columns) - set(live_columns))
        if uncomparable:
            raise PersistenceConflict(
                "INTERPRETATION_MIGRATION_AMBIGUOUS:"
                f"uncomparable_columns={','.join(uncomparable)}:kept_both_tables=true"
            )

        shared = ",".join(orphan_columns)
        live_rows = {
            row["interpretation_id"]: tuple(row)
            for row in conn.execute(f"SELECT {shared} FROM interpretations")
        }
        orphan_rows = {
            row["interpretation_id"]: tuple(row)
            for row in conn.execute(f"SELECT {shared} FROM interpretations_migrated")
        }

        absent = sorted(set(orphan_rows) - set(live_rows))
        differing = sorted(
            key for key, value in orphan_rows.items()
            if key in live_rows and live_rows[key] != value
        )
        if absent or differing:
            raise PersistenceConflict(
                "INTERPRETATION_MIGRATION_AMBIGUOUS:"
                f"orphan_records_absent_from_live={len(absent)}:"
                f"same_id_different_contents={len(differing)}:"
                f"live_rows={len(live_rows)}:orphan_rows={len(orphan_rows)}:"
                "kept_both_tables=true"
            )

    @staticmethod
    def _ensure_interpretation_binding(conn: sqlite3.Connection) -> None:
        """Add the interpretation binding columns to an older authorizations table.

        A database created before interpretations were persisted has no place to
        record which interpretation an approval was based on. Adding the columns
        keeps existing rows (with NULLs) readable, and a NULL binding is treated as
        "no interpretation" downstream, which fails closed rather than permitting an
        unbounded approval.
        """
        existing = {
            row["name"] for row in conn.execute("PRAGMA table_info(authorizations)").fetchall()
        }
        for column in ("interpretation_id", "interpretation_output_sha256"):
            if column not in existing:
                conn.execute(f"ALTER TABLE authorizations ADD COLUMN {column} TEXT")

    @staticmethod
    def _legacy_lifecycle_schema(conn: sqlite3.Connection) -> bool:
        for table, marker in _LEGACY_TABLE_MARKERS:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if row is not None and row["sql"] and marker in " ".join(row["sql"].split()):
                return True
        return False

    def _migrate_lifecycle_schema(self, conn: sqlite3.Connection) -> None:
        """Upgrade pre-MP-LIFECYCLE-006 databases in place, preserving all history.

        The legacy schema made payout_item_id globally unique and allowed exactly one
        reconciliation row per operation, which made routine asynchronous transitions
        unrecordable. Rebuilding both tables keeps every stored observation and
        reconciliation row byte-identical; only the uniqueness shape and the
        reconciliation surrogate key change.
        """
        if not self._legacy_lifecycle_schema(conn):
            return
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute("BEGIN IMMEDIATE")
            for trigger in (
                "observations_no_update",
                "observations_no_delete",
                "reconciliations_no_update",
                "reconciliations_no_delete",
            ):
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.execute("DROP TABLE IF EXISTS provider_observations_legacy")
            conn.execute("DROP TABLE IF EXISTS reconciliations_legacy")
            conn.execute("ALTER TABLE reconciliations RENAME TO reconciliations_legacy")
            conn.execute("ALTER TABLE provider_observations RENAME TO provider_observations_legacy")
            conn.execute(_PROVIDER_OBSERVATIONS_DDL)
            conn.execute(_RECONCILIATIONS_DDL)
            conn.execute(
                """INSERT INTO provider_observations(
                       observation_id,operation_key,payout_batch_id,payout_item_id,status,
                       recipient,currency,amount_cents,observed_at,raw_sha256
                   )
                   SELECT observation_id,operation_key,payout_batch_id,payout_item_id,status,
                          recipient,currency,amount_cents,observed_at,raw_sha256
                   FROM provider_observations_legacy"""
            )
            conn.execute(
                """INSERT INTO reconciliations(
                       operation_key,observation_id,outcome,reason_codes_json,reconciled_at
                   )
                   SELECT operation_key,observation_id,outcome,reason_codes_json,reconciled_at
                   FROM reconciliations_legacy"""
            )
            conn.execute("DROP TABLE reconciliations_legacy")
            conn.execute("DROP TABLE provider_observations_legacy")
            conn.execute(
                """CREATE TRIGGER observations_no_update
                   BEFORE UPDATE ON provider_observations
                   BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_PROVIDER_OBSERVATION'); END"""
            )
            conn.execute(
                """CREATE TRIGGER observations_no_delete
                   BEFORE DELETE ON provider_observations
                   BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_PROVIDER_OBSERVATION'); END"""
            )
            conn.execute(
                """CREATE TRIGGER reconciliations_no_update
                   BEFORE UPDATE ON reconciliations
                   BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_RECONCILIATION'); END"""
            )
            conn.execute(
                """CREATE TRIGGER reconciliations_no_delete
                   BEFORE DELETE ON reconciliations
                   BEGIN SELECT RAISE(ABORT, 'IMMUTABLE_RECONCILIATION'); END"""
            )
            self._append_event(
                conn,
                "schema:payment-lifecycle",
                "PAYOUT_LIFECYCLE_SCHEMA_MIGRATED",
                {"observations": "preserved", "reconciliations": "preserved"},
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.execute("PRAGMA foreign_keys = ON")

    def register_user(self, user_id: str, email: str, role: Role, password: str) -> None:
        if len(password) < 12:
            raise ValueError("password must contain at least 12 characters")
        password_hash = self.password_hasher.hash(password)
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO users(user_id,email,role,password_hash) VALUES(?,?,?,?)",
                (user_id, email, role.value, password_hash),
            )

    def login(self, email: str, password: str) -> AuthenticatedSession:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT user_id,role,password_hash FROM users WHERE email = ? COLLATE NOCASE",
                (email,),
            ).fetchone()
            if row is None:
                raise AuthenticationFailed("INVALID_CREDENTIALS")
            try:
                self.password_hasher.verify(row["password_hash"], password)
            except (VerifyMismatchError, InvalidHashError) as exc:
                raise AuthenticationFailed("INVALID_CREDENTIALS") from exc
            token = secrets.token_urlsafe(32)
            csrf_token = secrets.token_urlsafe(32)
            expires = datetime.now(timezone.utc) + timedelta(hours=8)
            conn.execute(
                "INSERT INTO sessions(token_hash,csrf_hash,user_id,role,expires_at) VALUES(?,?,?,?,?)",
                (
                    self._token_hash(token),
                    self._token_hash(csrf_token),
                    row["user_id"],
                    row["role"],
                    expires.isoformat(),
                ),
            )
        return AuthenticatedSession(token, csrf_token, row["user_id"], Role(row["role"]), expires)

    @staticmethod
    def _token_hash(token: str) -> str:
        return sha256(token.encode("utf-8")).hexdigest()

    def _require_session(
        self,
        conn: sqlite3.Connection,
        token: str,
        role: Role,
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT user_id,role,expires_at,revoked FROM sessions WHERE token_hash = ?",
            (self._token_hash(token),),
        ).fetchone()
        if row is None or row["revoked"]:
            raise AuthenticationFailed("SESSION_INVALID")
        if datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
            raise AuthenticationFailed("SESSION_EXPIRED")
        if row["role"] != role.value:
            raise AuthorizationDenied(f"{role.value.upper()}_ROLE_REQUIRED")
        return row

    def authenticated_session(self, token: str) -> AuthenticatedSession:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT user_id,role,expires_at,revoked FROM sessions WHERE token_hash=?",
                (self._token_hash(token),),
            ).fetchone()
            if row is None or row["revoked"]:
                raise AuthenticationFailed("SESSION_INVALID")
            expires = datetime.fromisoformat(row["expires_at"])
            if expires <= datetime.now(timezone.utc):
                raise AuthenticationFailed("SESSION_EXPIRED")
            return AuthenticatedSession(token, "", row["user_id"], Role(row["role"]), expires)

    def verify_csrf(self, token: str, csrf_token: str) -> None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT csrf_hash,expires_at,revoked FROM sessions WHERE token_hash=?",
                (self._token_hash(token),),
            ).fetchone()
            if (
                row is None
                or row["revoked"]
                or datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc)
                or not secrets.compare_digest(row["csrf_hash"], self._token_hash(csrf_token))
            ):
                raise AuthorizationDenied("CSRF_INVALID")

    def revoke_session(self, token: str) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE sessions SET revoked=1 WHERE token_hash=?",
                (self._token_hash(token),),
            )

    def user_email(self, user_id: str) -> str:
        with self._connection() as conn:
            row = conn.execute("SELECT email FROM users WHERE user_id=?", (user_id,)).fetchone()
            if row is None:
                raise PersistenceConflict("USER_NOT_FOUND")
            return row["email"]

    def load_fund_policy(self, fund_id: str, version: str) -> FundPolicy:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM fund_policies WHERE fund_id=? AND version=?",
                (fund_id, version),
            ).fetchone()
            if row is None:
                raise PersistenceConflict("FUND_POLICY_NOT_FOUND")
            return FundPolicy(
                fund_id=row["fund_id"],
                version=row["version"],
                allowed_activity_codes=frozenset(json.loads(row["allowed_activities_json"])),
                allowed_expense_categories=frozenset(json.loads(row["allowed_categories_json"])),
                starts_on=date.fromisoformat(row["starts_on"]),
                ends_on=date.fromisoformat(row["ends_on"]),
                available=FundBalance(row["currency"], row["available_cents"] - row["reserved_cents"]),
                receipt_required=bool(row["receipt_required"]),
            )

    def claim_snapshot(self, token: str, claim_id: str) -> dict[str, object]:
        with self._connection() as conn:
            session = conn.execute(
                "SELECT user_id,role,expires_at,revoked FROM sessions WHERE token_hash=?",
                (self._token_hash(token),),
            ).fetchone()
            if session is None or session["revoked"]:
                raise AuthenticationFailed("SESSION_INVALID")
            if datetime.fromisoformat(session["expires_at"]) <= datetime.now(timezone.utc):
                raise AuthenticationFailed("SESSION_EXPIRED")
            row = conn.execute(
                """SELECT c.*, e.media_type AS media_type, e.original_name AS original_name
                   FROM claims c LEFT JOIN evidence e ON e.sha256 = c.evidence_sha256
                   WHERE c.claim_id=? ORDER BY c.revision DESC LIMIT 1""",
                (claim_id,),
            ).fetchone()
            if row is None:
                raise PersistenceConflict("CLAIM_NOT_FOUND")
            if session["role"] == Role.VOLUNTEER.value and row["volunteer_id"] != session["user_id"]:
                raise AuthorizationDenied("CLAIM_OWNER_REQUIRED")
            decisions = conn.execute(
                """SELECT fund_id,policy_version,outcome,reason_codes_json
                   FROM policy_decisions WHERE claim_id=? AND revision=? ORDER BY fund_id""",
                (claim_id, row["revision"]),
            ).fetchall()
            authority = conn.execute(
                """SELECT authorization_id,reviewer_id FROM authorizations
                   WHERE claim_id=? AND claim_revision=?""",
                (claim_id, row["revision"]),
            ).fetchone()
            operation = conn.execute(
                """SELECT o.operation_key,o.state FROM payment_intents i
                   JOIN payment_operations o ON o.operation_key=i.operation_key
                   WHERE i.claim_id=? AND i.claim_revision=?""",
                (claim_id, row["revision"]),
            ).fetchone()
            return {
                "claim": self._claim_from_row(row),
                "revision": row["revision"],
                "decisions": [
                    {
                        "fund_id": item["fund_id"],
                        "policy_version": item["policy_version"],
                        "outcome": item["outcome"],
                        "reason_codes": tuple(json.loads(item["reason_codes_json"])),
                    }
                    for item in decisions
                ],
                "authorization_id": authority["authorization_id"] if authority else None,
                "reviewer_id": authority["reviewer_id"] if authority else None,
                "operation_key": operation["operation_key"] if operation else None,
                "operation_state": operation["state"] if operation else None,
            }

    def review_queue(self, token: str) -> list[dict[str, object]]:
        with self._connection() as conn:
            self._require_session(conn, token, Role.REVIEWER)
            rows = conn.execute(
                """SELECT c.claim_id,c.revision,c.amount_cents,c.currency,c.recipient
                   FROM claims c JOIN policy_decisions d
                     ON d.claim_id=c.claim_id AND d.revision=c.revision
                   WHERE d.outcome='eligible'
                     AND c.revision=(SELECT MAX(c2.revision) FROM claims c2 WHERE c2.claim_id=c.claim_id)
                   ORDER BY c.claim_id"""
            ).fetchall()
            return [dict(row) for row in rows]

    def existing_authorization_id(self, claim_id: str, revision: int) -> str | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT authorization_id FROM authorizations WHERE claim_id=? AND claim_revision=?",
                (claim_id, revision),
            ).fetchone()
            return row["authorization_id"] if row else None

    def existing_operation_for_claim(self, claim_id: str, revision: int) -> str | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT operation_key FROM payment_intents WHERE claim_id=? AND claim_revision=?",
                (claim_id, revision),
            ).fetchone()
            return row["operation_key"] if row else None

    def create_fund(self, policy: FundPolicy) -> None:
        if policy.available.cents <= 0:
            # Zero *available* is a legitimate post-reservation state; a fund is never
            # opened with nothing to spend, and that refusal is explicit here.
            raise ValueError("FUND_ALLOCATION_MUST_BE_POSITIVE")
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO fund_policies(
                    fund_id,version,allowed_activities_json,allowed_categories_json,
                    starts_on,ends_on,currency,available_cents,receipt_required
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    policy.fund_id,
                    policy.version,
                    json.dumps(sorted(policy.allowed_activity_codes)),
                    json.dumps(sorted(policy.allowed_expense_categories)),
                    policy.starts_on.isoformat(),
                    policy.ends_on.isoformat(),
                    policy.available.currency,
                    policy.available.cents,
                    int(policy.receipt_required),
                ),
            )

    def submit_claim(self, token: str, claim: Claim, *, revision: int) -> None:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                session = self._require_session(conn, token, Role.VOLUNTEER)
                if session["user_id"] != claim.volunteer_id:
                    raise AuthorizationDenied("VOLUNTEER_ID_MISMATCH")
                prior = conn.execute(
                    "SELECT volunteer_id,MAX(revision) AS latest FROM claims WHERE claim_id = ?",
                    (claim.claim_id,),
                ).fetchone()
                if prior["latest"] is not None:
                    if prior["volunteer_id"] != claim.volunteer_id:
                        raise AuthorizationDenied("CLAIM_OWNER_MISMATCH")
                    if revision != prior["latest"] + 1:
                        raise PersistenceConflict("CLAIM_REVISION_NOT_SEQUENTIAL")
                elif revision != 1:
                    raise PersistenceConflict("CLAIM_REVISION_MUST_START_AT_ONE")
                conn.execute(
                    "INSERT OR IGNORE INTO evidence(sha256,media_type,original_name) VALUES(?,?,?)",
                    (claim.evidence.sha256, claim.evidence.media_type, claim.evidence.original_name),
                )
                conn.execute(
                    """INSERT INTO claims(
                        claim_id,revision,volunteer_id,activity_id,activity_code,occurred_on,
                        evidence_sha256,currency,amount_cents,recipient,expense_category
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        claim.claim_id,
                        revision,
                        claim.volunteer_id,
                        claim.activity.activity_id,
                        claim.activity.activity_code,
                        claim.activity.occurred_on.isoformat(),
                        claim.evidence.sha256,
                        claim.requested.currency,
                        claim.requested.cents,
                        claim.recipient,
                        claim.expense_category,
                    ),
                )
                self._append_event(conn, claim.claim_id, "CLAIM_SUBMITTED", {"revision": revision})
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def record_policy_decision(self, decision: PolicyDecision, *, revision: int) -> None:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT revision FROM claims WHERE claim_id = ? AND revision = ?",
                    (decision.claim_id, revision),
                ).fetchone()
                if row is None:
                    raise PersistenceConflict("CLAIM_NOT_FOUND")
                conn.execute(
                    """INSERT INTO policy_decisions(
                        claim_id,revision,fund_id,policy_version,outcome,reason_codes_json
                    ) VALUES(?,?,?,?,?,?)""",
                    (
                        decision.claim_id,
                        revision,
                        decision.fund_id,
                        decision.policy_version,
                        decision.outcome.value,
                        json.dumps(decision.reason_codes),
                    ),
                )
                self._append_event(
                    conn,
                    decision.claim_id,
                    "POLICY_DECIDED",
                    {"revision": revision, "outcome": decision.outcome.value},
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    # -- interpretation lifecycle ---------------------------------------

    def record_interpretation(
        self,
        *,
        claim_id: str,
        revision: int,
        evidence_sha256: str,
        ai_mode: str,
        model_id: str,
        config_sha256: str,
        status: str,
        output_sha256: str,
        facts: "list[dict[str, object]] | tuple[()]" = (),
        missing: tuple[str, ...] = (),
        error_code: str | None = None,
        contract_version: str | None = None,
    ) -> dict[str, object]:
        """Persist one interpretation attempt, idempotently.

        The natural key is (claim, revision, evidence digest, ai mode, model id,
        config digest), so re-running the same configuration against the same
        evidence reuses the stored record instead of creating a second one. That is
        what makes a page refresh free of fresh inference.

        A REFUSED attempt is recorded too: a refusal is evidence, and its absence
        would be indistinguishable from "never attempted".
        """
        if status not in {"ACCEPTED", "REFUSED"}:
            raise ValueError("INTERPRETATION_STATUS_INVALID")
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # Post-approval immutability lives here, not only in the route, so a
                # direct store call cannot attach a new interpretation to a revision a
                # reviewer already approved.
                authorized = conn.execute(
                    "SELECT 1 FROM authorizations WHERE claim_id=? AND claim_revision=?",
                    (claim_id, revision),
                ).fetchone()
                if authorized is not None:
                    raise PersistenceConflict("INTERPRETATION_AFTER_AUTHORIZATION")

                # An existing ACCEPTED result for this exact configuration is reused
                # rather than re-run: the caller must not pay for inference whose
                # answer is already on record.
                accepted = conn.execute(
                    """SELECT * FROM interpretations
                       WHERE claim_id=? AND claim_revision=? AND evidence_sha256=?
                         AND ai_mode=? AND model_id=? AND config_sha256=?
                         AND status='ACCEPTED'
                       ORDER BY attempt DESC LIMIT 1""",
                    (claim_id, revision, evidence_sha256, ai_mode, model_id, config_sha256),
                ).fetchone()
                if accepted is not None:
                    conn.execute("COMMIT")
                    return {**dict(accepted), "reused": True}

                # A previous REFUSED attempt does not occupy the key: retrying after a
                # failure records attempt N+1 and preserves the refusal as history.
                highest = conn.execute(
                    """SELECT COALESCE(MAX(attempt), 0) AS last FROM interpretations
                       WHERE claim_id=? AND claim_revision=? AND evidence_sha256=?
                         AND ai_mode=? AND model_id=? AND config_sha256=?""",
                    (claim_id, revision, evidence_sha256, ai_mode, model_id, config_sha256),
                ).fetchone()["last"]
                attempt = int(highest) + 1

                interpretation_id = f"interp-{uuid4().hex}"
                conn.execute(
                    """INSERT INTO interpretations(
                        interpretation_id,claim_id,claim_revision,evidence_sha256,ai_mode,
                        model_id,config_sha256,attempt,status,error_code,output_sha256,
                        facts_json,missing_json,contract_version,created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        interpretation_id,
                        claim_id,
                        revision,
                        evidence_sha256,
                        ai_mode,
                        model_id,
                        config_sha256,
                        attempt,
                        status,
                        error_code,
                        output_sha256,
                        json.dumps(list(facts), sort_keys=True, separators=(",", ":")),
                        json.dumps(list(missing), sort_keys=True, separators=(",", ":")),
                        contract_version,
                        created_at,
                    ),
                )
                self._append_event(
                    conn,
                    claim_id,
                    "EVIDENCE_INTERPRETED",
                    {
                        "interpretation_id": interpretation_id,
                        "revision": revision,
                        "attempt": attempt,
                        "status": status,
                        "model_id": model_id,
                        "ai_mode": ai_mode,
                        "output_sha256": output_sha256,
                    },
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return {
            "interpretation_id": interpretation_id,
            "claim_id": claim_id,
            "claim_revision": revision,
            "evidence_sha256": evidence_sha256,
            "ai_mode": ai_mode,
            "model_id": model_id,
            "config_sha256": config_sha256,
            "attempt": attempt,
            "status": status,
            "error_code": error_code,
            "output_sha256": output_sha256,
            "facts_json": json.dumps(list(facts), sort_keys=True, separators=(",", ":")),
            "missing_json": json.dumps(list(missing), sort_keys=True, separators=(",", ":")),
            "contract_version": contract_version,
            "created_at": created_at,
            "reused": False,
        }

    def accepted_interpretation(
        self, claim_id: str, revision: int, evidence_sha256: str | None = None
    ) -> dict[str, object] | None:
        """The current accepted interpretation for this claim revision.

        Uses the single ordering rule in :func:`_latest_accepted`, so this is the same
        row the approval, intent and dispatch paths will select.
        """
        with self._connection() as conn:
            row = _latest_accepted(conn, claim_id, revision, evidence_sha256)
            return dict(row) if row is not None else None

    def accepted_interpretation_for_config(
        self,
        claim_id: str,
        revision: int,
        evidence_sha256: str,
        ai_mode: str,
        model_id: str,
        config_sha256: str,
    ) -> dict[str, object] | None:
        """The accepted result for *this exact configuration*, if one exists.

        Reuse must be decided on the configuration that is about to run, not on
        "some interpretation exists": switching mode or model has to produce a new
        result rather than silently serving the previous one.
        """
        with self._connection() as conn:
            row = conn.execute(
                f"""SELECT * FROM interpretations
                    WHERE claim_id=? AND claim_revision=? AND evidence_sha256=?
                      AND ai_mode=? AND model_id=? AND config_sha256=? AND status='ACCEPTED'
                    {_ATTEMPT_ORDER} LIMIT 1""",
                (claim_id, revision, evidence_sha256, ai_mode, model_id, config_sha256),
            ).fetchone()
            return dict(row) if row is not None else None

    def interpretation_history(self, claim_id: str, revision: int) -> list[dict[str, object]]:
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT * FROM interpretations WHERE claim_id=? AND claim_revision=?
                   ORDER BY created_at, interpretation_id""",
                (claim_id, revision),
            ).fetchall()
        return [dict(row) for row in rows]

    def authorization_binding(self, operation_key: str) -> dict[str, object]:
        """Everything an operator needs to re-verify authority at dispatch time.

        Deliberately returns the binding columns as ``None`` for a legacy row rather
        than substituting a default: an authorization created before interpretations
        were bound has no reviewed output, and pretending otherwise is exactly the
        silent retrofit this returns information to prevent.
        """
        with self._connection() as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(authorizations)")}
            binding_columns = (
                "a.interpretation_id,a.interpretation_output_sha256"
                if "interpretation_id" in columns
                else "NULL AS interpretation_id,NULL AS interpretation_output_sha256"
            )
            row = conn.execute(
                f"""SELECT a.authorization_id,a.reviewer_id,a.claim_id,a.claim_revision,
                          a.evidence_sha256,a.fund_id,a.policy_version,a.recipient,
                          a.currency,a.amount_cents,
                          {binding_columns},
                          i.claim_id AS intent_claim_id,
                          i.claim_revision AS intent_claim_revision,
                          i.evidence_sha256 AS intent_evidence_sha256,
                          i.fund_id AS intent_fund_id,
                          i.policy_version AS intent_policy_version,
                          i.recipient AS intent_recipient,
                          i.currency AS intent_currency,
                          i.amount_cents AS intent_amount_cents
                   FROM payment_intents i
                   JOIN authorizations a ON a.authorization_id = i.authorization_id
                   WHERE i.operation_key=?""",
                (operation_key,),
            ).fetchone()
            if row is None:
                raise PersistenceConflict("AUTHORIZATION_NOT_FOUND")
            return dict(row)

    def accepted_interpretation_matches(
        self, claim_id: str, revision: int, evidence_sha256: str,
        interpretation_id: str | None, output_sha256: str | None,
    ) -> bool:
        """True only if the named interpretation is still the accepted one for this evidence."""
        if not interpretation_id or not output_sha256:
            return False
        with self._connection() as conn:
            current = _latest_accepted(conn, claim_id, revision, evidence_sha256)
        if current is None:
            return False
        return (
            _constant_time_equal(current["interpretation_id"], interpretation_id)
            and _constant_time_equal(current["output_sha256"], output_sha256)
        )

    # -- hosted qualification attempts ------------------------------------

    def claim_by_id(self, claim_id: str, revision: int | None = None) -> dict[str, object]:
        """Read a claim without a session.

        Only the operator-level qualification command uses this. The web application
        always goes through :meth:`claim_snapshot`, which authenticates a session and
        checks ownership; this method exists because a CLI process has no cookie to
        present, and it is deliberately not reachable from a route.
        """
        with self._connection() as conn:
            if revision is None:
                row = conn.execute(
                    """SELECT c.*, e.media_type AS media_type, e.original_name AS original_name
                       FROM claims c LEFT JOIN evidence e ON e.sha256 = c.evidence_sha256
                       WHERE c.claim_id=? ORDER BY c.revision DESC LIMIT 1""",
                    (claim_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT c.*, e.media_type AS media_type, e.original_name AS original_name
                       FROM claims c LEFT JOIN evidence e ON e.sha256 = c.evidence_sha256
                       WHERE c.claim_id=? AND c.revision=?""",
                    (claim_id, revision),
                ).fetchone()
        if row is None:
            raise PersistenceConflict("CLAIM_NOT_FOUND")
        return {"claim": self._claim_from_row(row), "revision": int(row["revision"])}

    def record_hosted_attempt(
        self,
        *,
        authorization_id: str,
        claim_id: str,
        claim_revision: int,
        evidence_sha256: str,
        expectation_sha256: str,
        model_id: str,
        config_sha256: str,
        max_output_tokens: int,
        max_total_tokens: int,
    ) -> dict[str, object]:
        """Claim one authorization for one send. Committed before the request leaves.

        The primary key is the authorization reference, so the second attempt under the
        same authorization is refused by the database rather than by a flag in the
        process. That is what survives a crash, a timeout or a restart: the row exists
        whether or not the process that wrote it lived to see the answer.
        """
        recorded_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as conn:
            try:
                conn.execute(
                    """INSERT INTO hosted_attempts(
                        authorization_id,claim_id,claim_revision,evidence_sha256,
                        expectation_sha256,model_id,config_sha256,max_output_tokens,
                        max_total_tokens,state,recorded_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,'RECORDED',?)""",
                    (
                        authorization_id,
                        claim_id,
                        claim_revision,
                        evidence_sha256,
                        expectation_sha256,
                        model_id,
                        config_sha256,
                        max_output_tokens,
                        max_total_tokens,
                        recorded_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise PersistenceConflict("HOSTED_ATTEMPT_ALREADY_RECORDED") from exc
        return {
            "authorization_id": authorization_id,
            "claim_id": claim_id,
            "claim_revision": claim_revision,
            "evidence_sha256": evidence_sha256,
            "expectation_sha256": expectation_sha256,
            "model_id": model_id,
            "config_sha256": config_sha256,
            "max_output_tokens": max_output_tokens,
            "max_total_tokens": max_total_tokens,
            "state": "RECORDED",
            "recorded_at": recorded_at,
        }

    def finish_hosted_attempt(
        self,
        authorization_id: str,
        *,
        state: str,
        code: str | None = None,
        usage: dict[str, int] | None = None,
        diagnostic: dict[str, object] | None = None,
    ) -> None:
        """Move a RECORDED attempt to a terminal state, once.

        ``WHERE state='RECORDED'`` makes a second finish a no-op failure rather than an
        overwrite: an outcome that has been written down is evidence, and evidence this
        system does not rewrite.
        """
        if state not in _HOSTED_ATTEMPT_TERMINAL:
            raise PersistenceConflict("HOSTED_ATTEMPT_STATE_INVALID")
        usage_json = (
            json.dumps(usage, sort_keys=True, separators=(",", ":")) if usage else None
        )
        cleaned = clean_diagnostic(diagnostic)
        diagnostic_json = (
            json.dumps(cleaned, sort_keys=True, separators=(",", ":")) if cleaned else None
        )
        with self._connection() as conn:
            cursor = conn.execute(
                """UPDATE hosted_attempts
                   SET state=?,code=?,usage_json=?,diagnostic_json=?,finished_at=?
                   WHERE authorization_id=? AND state='RECORDED'""",
                (
                    state,
                    code,
                    usage_json,
                    diagnostic_json,
                    datetime.now(timezone.utc).isoformat(),
                    authorization_id,
                ),
            )
            if cursor.rowcount != 1:
                raise PersistenceConflict("HOSTED_ATTEMPT_NOT_PENDING")

    def hosted_attempt(self, authorization_id: str) -> dict[str, object] | None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM hosted_attempts WHERE authorization_id=?",
                (authorization_id,),
            ).fetchone()
        return None if row is None else _hosted_attempt_row(row)

    def hosted_attempts(self) -> list[dict[str, object]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM hosted_attempts ORDER BY recorded_at"
            ).fetchall()
        return [_hosted_attempt_row(row) for row in rows]

    def record_interpretation_usage(
        self, claim_id: str, revision: int, usage: dict[str, int], *, model_id: str
    ) -> None:
        """Append the provider's token counters to the audit trail.

        Usage is evidence about what an accepted interpretation cost, so it is recorded
        next to the interpretation rather than printed and forgotten.
        """
        with self._connection() as conn:
            self._append_event(
                conn,
                claim_id,
                "INTERPRETATION_USAGE",
                {
                    "claim_id": claim_id,
                    "revision": revision,
                    "model_id": model_id,
                    "usage": usage,
                },
            )

    def record_citation_refusal(
        self,
        claim_id: str,
        revision: int,
        diagnostic: dict[str, object],
        *,
        model_id: str,
    ) -> None:
        """Append a citation refusal's shape to the audit trail.

        The payload passes through :func:`clean_diagnostic`, so it can hold a fact name,
        a category and offsets — never the quote, the receipt or the key.
        """
        cleaned = clean_diagnostic(diagnostic)
        if not cleaned:
            return
        with self._connection() as conn:
            self._append_event(
                conn,
                claim_id,
                "INTERPRETATION_CITATION_REFUSAL",
                {
                    "claim_id": claim_id,
                    "revision": revision,
                    "model_id": model_id,
                    "diagnostic": cleaned,
                },
            )

    def authorize_claim(
        self,
        token: str,
        claim_id: str,
        *,
        revision: int,
        interpretation_id: str,
        interpretation_output_sha256: str,
    ) -> HumanAuthorization:
        """Freeze an approval against the exact interpretation the reviewer saw.

        The caller must name the interpretation and output digest that were rendered on
        the reviewer's page. If anything newer has been accepted since, the page is
        stale and the approval is refused rather than silently upgraded to output the
        reviewer never saw.
        """
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                session = self._require_session(conn, token, Role.REVIEWER)
                rows = conn.execute(
                    """SELECT c.*,d.fund_id,d.policy_version,d.outcome
                       FROM claims c JOIN policy_decisions d
                         ON d.claim_id=c.claim_id AND d.revision=c.revision
                       WHERE c.claim_id=? AND c.revision=? AND d.outcome=?""",
                    (claim_id, revision, PolicyOutcome.ELIGIBLE.value),
                ).fetchall()
                if len(rows) != 1:
                    raise AuthorizationDenied("ONE_ELIGIBLE_DECISION_REQUIRED")
                row = rows[0]
                if row["volunteer_id"] == session["user_id"]:
                    raise AuthorizationDenied("SELF_APPROVAL_FORBIDDEN")

                # The AI stage is a precondition of review, and the reviewer must have
                # seen *this* output: same claim, revision and evidence digest.
                supplied = conn.execute(
                    """SELECT interpretation_id,output_sha256,status,evidence_sha256
                       FROM interpretations
                       WHERE interpretation_id=? AND claim_id=? AND claim_revision=?""",
                    (interpretation_id, claim_id, revision),
                ).fetchone()
                if supplied is None:
                    raise AuthorizationDenied("INTERPRETATION_NOT_FOUND")
                if supplied["status"] != "ACCEPTED":
                    raise AuthorizationDenied("INTERPRETATION_NOT_ACCEPTED")
                if supplied["evidence_sha256"] != row["evidence_sha256"]:
                    raise AuthorizationDenied("INTERPRETATION_EVIDENCE_MISMATCH")
                if not _constant_time_equal(
                    supplied["output_sha256"], interpretation_output_sha256
                ):
                    raise AuthorizationDenied("INTERPRETATION_OUTPUT_MISMATCH")

                # A newer accepted result exists: the reviewer's page is stale. This
                # uses the same selection rule as the page and the dispatch check.
                current = _latest_accepted(conn, claim_id, revision, row["evidence_sha256"])
                if current is None:
                    raise AuthorizationDenied("ACCEPTED_INTERPRETATION_REQUIRED")
                if current["interpretation_id"] != supplied["interpretation_id"]:
                    raise AuthorizationDenied("INTERPRETATION_STALE")

                authorization_id = f"auth-{uuid4().hex}"
                created_at = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    """INSERT INTO authorizations(
                        authorization_id,reviewer_id,claim_id,claim_revision,evidence_sha256,
                        fund_id,policy_version,recipient,currency,amount_cents,created_at,
                        interpretation_id,interpretation_output_sha256
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        authorization_id,
                        session["user_id"],
                        claim_id,
                        revision,
                        row["evidence_sha256"],
                        row["fund_id"],
                        row["policy_version"],
                        row["recipient"],
                        row["currency"],
                        row["amount_cents"],
                        created_at,
                        supplied["interpretation_id"],
                        supplied["output_sha256"],
                    ),
                )
                self._append_event(
                    conn,
                    claim_id,
                    "CLAIM_AUTHORIZED",
                    {
                        "authorization_id": authorization_id,
                        "revision": revision,
                        "interpretation_id": supplied["interpretation_id"],
                        "interpretation_output_sha256": supplied["output_sha256"],
                    },
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return HumanAuthorization(
            authorization_id=authorization_id,
            reviewer_id=session["user_id"],
            reviewer_role="REIMBURSEMENT_REVIEWER",
            authenticated=True,
            claim_id=claim_id,
            claim_revision=revision,
            evidence_sha256=row["evidence_sha256"],
            policy_version=row["policy_version"],
            fund_id=row["fund_id"],
            recipient=row["recipient"],
            amount=Money(row["currency"], row["amount_cents"]),
        )

    def create_payment_intent(self, authorization_id: str) -> PaymentIntent:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM authorizations WHERE authorization_id = ?",
                    (authorization_id,),
                ).fetchone()
                if row is None:
                    raise AuthorizationDenied("AUTHORIZATION_NOT_FOUND")
                latest = conn.execute(
                    "SELECT MAX(revision) AS revision FROM claims WHERE claim_id = ?",
                    (row["claim_id"],),
                ).fetchone()["revision"]
                if latest != row["claim_revision"]:
                    raise StaleAuthorization("CLAIM_REVISION_CHANGED")

                # An approval is only valid for the interpretation it was reviewed
                # against. If a materially different interpretation has since been
                # accepted, the reviewer saw something else and the approval is stale.
                accepted = _latest_accepted(
                    conn, row["claim_id"], row["claim_revision"]
                )
                if accepted is None:
                    raise StaleAuthorization("INTERPRETATION_MISSING")
                if row["interpretation_output_sha256"] is None:
                    raise StaleAuthorization("INTERPRETATION_BINDING_MISSING")
                if (
                    row["interpretation_id"] != accepted["interpretation_id"]
                    or row["interpretation_output_sha256"] != accepted["output_sha256"]
                ):
                    raise StaleAuthorization("INTERPRETATION_CHANGED")
                existing = conn.execute(
                    "SELECT * FROM payment_intents WHERE authorization_id = ?",
                    (authorization_id,),
                ).fetchone()
                if existing is not None:
                    intent = self._intent_from_row(existing)
                    conn.execute("COMMIT")
                    return intent
                claim_row = conn.execute(
                    """SELECT c.*, e.media_type AS media_type, e.original_name AS original_name
                       FROM claims c LEFT JOIN evidence e ON e.sha256 = c.evidence_sha256
                       WHERE c.claim_id=? AND c.revision=?""",
                    (row["claim_id"], row["claim_revision"]),
                ).fetchone()
                claim = self._claim_from_row(claim_row)
                decision = PolicyDecision(
                    claim_id=row["claim_id"],
                    fund_id=row["fund_id"],
                    policy_version=row["policy_version"],
                    outcome=PolicyOutcome.ELIGIBLE,
                    reason_codes=("ELIGIBLE",),
                )
                authorization = HumanAuthorization(
                    authorization_id=row["authorization_id"],
                    reviewer_id=row["reviewer_id"],
                    reviewer_role="REIMBURSEMENT_REVIEWER",
                    authenticated=True,
                    claim_id=row["claim_id"],
                    claim_revision=row["claim_revision"],
                    evidence_sha256=row["evidence_sha256"],
                    policy_version=row["policy_version"],
                    fund_id=row["fund_id"],
                    recipient=row["recipient"],
                    amount=Money(row["currency"], row["amount_cents"]),
                )
                intent = bind_payment_intent(claim, decision, authorization)
                reserved = conn.execute(
                    """UPDATE fund_policies
                       SET reserved_cents = reserved_cents + ?
                       WHERE fund_id=? AND version=?
                         AND available_cents - reserved_cents >= ?""",
                    (
                        intent.amount.cents,
                        intent.fund_id,
                        intent.policy_version,
                        intent.amount.cents,
                    ),
                )
                if reserved.rowcount != 1:
                    raise InsufficientFunds("INSUFFICIENT_UNRESERVED_FUNDS")
                conn.execute(
                    """INSERT INTO payment_intents(
                        operation_key,authorization_id,claim_id,claim_revision,evidence_sha256,
                        fund_id,policy_version,recipient,currency,amount_cents,created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        intent.operation_key,
                        intent.authorization_id,
                        intent.claim_id,
                        intent.claim_revision,
                        intent.evidence_sha256,
                        intent.fund_id,
                        intent.policy_version,
                        intent.recipient,
                        intent.amount.currency,
                        intent.amount.cents,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                digest = intent.operation_key.removeprefix("missionpay:")
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    """INSERT INTO payment_operations(
                        operation_key,request_id,sender_batch_id,state,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?)""",
                    (
                        intent.operation_key,
                        f"mp-{digest[:35]}",
                        f"mp-{digest[:27]}",
                        "PREPARED",
                        now,
                        now,
                    ),
                )
                self._append_event(
                    conn,
                    intent.claim_id,
                    "PAYMENT_INTENT_CREATED",
                    {"operation_key": intent.operation_key},
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return intent

    def load_intent(self, operation_key: str) -> PaymentIntent:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM payment_intents WHERE operation_key=?",
                (operation_key,),
            ).fetchone()
            if row is None:
                raise PersistenceConflict("PAYMENT_INTENT_NOT_FOUND")
            return self._intent_from_row(row)

    def mark_recovery_conflict(self, operation_key: str, detail: str) -> None:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                changed = conn.execute(
                    """UPDATE payment_operations SET state='CONFLICT',detail=?,updated_at=?
                       WHERE operation_key=? AND state='UNCERTAIN'""",
                    (detail, datetime.now(timezone.utc).isoformat(), operation_key),
                )
                if changed.rowcount != 1:
                    raise PersistenceConflict("UNCERTAIN_OPERATION_REQUIRED")
                self._append_event(
                    conn,
                    operation_key,
                    "PAYOUT_RECOVERY_CONFLICT",
                    {"detail": detail},
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def fund_totals(self, fund_id: str, version: str) -> tuple[int, int]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT available_cents,reserved_cents FROM fund_policies WHERE fund_id=? AND version=?",
                (fund_id, version),
            ).fetchone()
            if row is None:
                raise PersistenceConflict("FUND_NOT_FOUND")
            return row["available_cents"], row["reserved_cents"]

    def begin_dispatch(self, operation_key: str, owner_id: str) -> None:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT state FROM payment_operations WHERE operation_key=?",
                    (operation_key,),
                ).fetchone()
                if row is None:
                    raise DispatchBlocked("OPERATION_NOT_FOUND")
                if row["state"] == "UNCERTAIN":
                    raise DispatchBlocked("UNCERTAIN_REQUIRES_RECONCILIATION")
                changed = conn.execute(
                    """UPDATE payment_operations
                       SET state='SENDING',owner_id=?,updated_at=?
                       WHERE operation_key=? AND state='PREPARED'""",
                    (owner_id, datetime.now(timezone.utc).isoformat(), operation_key),
                )
                if changed.rowcount != 1:
                    raise DispatchBlocked("DISPATCH_ALREADY_CLAIMED")
                self._append_event(
                    conn,
                    operation_key,
                    "PAYOUT_DISPATCH_STARTED",
                    {"owner_id": owner_id},
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def mark_dispatch_uncertain(self, operation_key: str, owner_id: str, detail: str) -> None:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                changed = conn.execute(
                    """UPDATE payment_operations
                       SET state='UNCERTAIN',detail=?,updated_at=?
                       WHERE operation_key=? AND state='SENDING' AND owner_id=?""",
                    (detail, datetime.now(timezone.utc).isoformat(), operation_key, owner_id),
                )
                if changed.rowcount != 1:
                    raise DispatchBlocked("DISPATCH_OWNER_OR_STATE_MISMATCH")
                self._append_event(
                    conn,
                    operation_key,
                    "PAYOUT_DISPATCH_UNCERTAIN",
                    {"detail": detail},
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def record_payout_acknowledgement(self, operation_key: str, payout_batch_id: str) -> None:
        if not payout_batch_id or not payout_batch_id.replace("-", "").replace("_", "").isalnum():
            raise PersistenceConflict("PAYOUT_BATCH_ID_INVALID")
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT state,payout_batch_id FROM payment_operations WHERE operation_key=?",
                    (operation_key,),
                ).fetchone()
                if row is None:
                    raise PersistenceConflict("OPERATION_NOT_FOUND")
                if row["state"] == "ACKNOWLEDGED" and row["payout_batch_id"] == payout_batch_id:
                    conn.execute("COMMIT")
                    return
                if row["state"] not in {"SENDING", "UNCERTAIN"}:
                    raise DispatchBlocked("ACKNOWLEDGEMENT_STATE_INVALID")
                conn.execute(
                    """UPDATE payment_operations
                       SET state='ACKNOWLEDGED',payout_batch_id=?,updated_at=?
                       WHERE operation_key=?""",
                    (payout_batch_id, datetime.now(timezone.utc).isoformat(), operation_key),
                )
                self._append_event(
                    conn,
                    operation_key,
                    "PAYOUT_ACKNOWLEDGED",
                    {"payout_batch_id": payout_batch_id},
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def persist_reconciliation(
        self,
        operation_key: str,
        observation: ProviderObservation,
        reconciliation: Reconciliation,
    ) -> bool:
        """Append one authenticated provider observation and re-derive operation state.

        Observations and reconciliations are append-only: a later readback of the same
        provider item is a new row, never an overwrite, and the operation's derived
        state always comes from the full ordered history. Returns False when the exact
        same provider evidence was already recorded (idempotent replay).
        """
        if reconciliation.operation_key != operation_key:
            raise PersistenceConflict("RECONCILIATION_OPERATION_MISMATCH")
        if reconciliation.observation_id != observation.observation_id:
            raise PersistenceConflict("RECONCILIATION_OBSERVATION_MISMATCH")
        if observation.attempt_id != operation_key:
            raise PersistenceConflict("OBSERVATION_ATTEMPT_MISMATCH")
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                operation = conn.execute(
                    """SELECT state,payout_batch_id,payout_item_id
                       FROM payment_operations WHERE operation_key=?""",
                    (operation_key,),
                ).fetchone()
                if operation is None:
                    raise PersistenceConflict("OPERATION_NOT_FOUND")
                if operation["state"] not in OBSERVABLE_OPERATION_STATES:
                    # UNCERTAIN must bind an authoritative batch first; PREPARED and
                    # SENDING have no acknowledged payout to read back.
                    raise PersistenceConflict("ACKNOWLEDGED_OPERATION_REQUIRED")
                if operation["payout_batch_id"] != observation.payout_batch_id:
                    raise PersistenceConflict("PAYOUT_BATCH_MISMATCH")
                replay = conn.execute(
                    """SELECT observation_id FROM provider_observations
                       WHERE operation_key=? AND payout_item_id=? AND raw_sha256=?""",
                    (operation_key, observation.payout_item_id, observation.raw_sha256),
                ).fetchone()
                if replay is not None:
                    conn.execute("COMMIT")
                    return False
                bound_item = operation["payout_item_id"]
                effective = reconciliation
                if bound_item is not None and bound_item != observation.payout_item_id:
                    # The evidence is retained, but an operation already bound to one
                    # provider item can never treat a second item as its own truth.
                    reasons: tuple[str, ...] = ("PAYOUT_ITEM_ID_MISMATCH",)
                    if reconciliation.outcome is ReconciliationOutcome.MISMATCH:
                        reasons += reconciliation.reason_codes
                    effective = Reconciliation(
                        operation_key=operation_key,
                        observation_id=observation.observation_id,
                        outcome=ReconciliationOutcome.MISMATCH,
                        reason_codes=reasons,
                    )
                conn.execute(
                    """INSERT INTO provider_observations(
                        observation_id,operation_key,payout_batch_id,payout_item_id,status,
                        recipient,currency,amount_cents,observed_at,raw_sha256
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        observation.observation_id,
                        operation_key,
                        observation.payout_batch_id,
                        observation.payout_item_id,
                        observation.status.value,
                        observation.recipient,
                        observation.amount.currency,
                        observation.amount.cents,
                        observation.observed_at.isoformat(),
                        observation.raw_sha256,
                    ),
                )
                conn.execute(
                    """INSERT INTO reconciliations(
                        operation_key,observation_id,outcome,reason_codes_json,reconciled_at
                    ) VALUES(?,?,?,?,?)""",
                    (
                        operation_key,
                        observation.observation_id,
                        effective.outcome.value,
                        json.dumps(effective.reason_codes),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                outcome, states = self._latest_item_states(conn, operation_key)
                if outcome is None:  # pragma: no cover - the inserted row always yields one
                    raise PersistenceConflict("RECONCILIATION_DERIVATION_FAILED")
                state = _OPERATION_STATE_BY_OUTCOME[outcome]
                conn.execute(
                    """UPDATE payment_operations
                       SET state=?,payout_item_id=COALESCE(payout_item_id,?),updated_at=?
                       WHERE operation_key=?""",
                    (
                        state,
                        observation.payout_item_id,
                        datetime.now(timezone.utc).isoformat(),
                        operation_key,
                    ),
                )
                self._append_event(
                    conn,
                    operation_key,
                    "PAYOUT_RECONCILED",
                    {
                        "outcome": effective.outcome.value,
                        "observation_id": observation.observation_id,
                        "payout_item_id": observation.payout_item_id,
                        "provider_status": observation.status.value,
                        "operation_state": state,
                        "observed_items": len(states),
                    },
                )
                conn.execute("COMMIT")
                return True
            except Exception:
                conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _latest_item_states(
        conn: sqlite3.Connection,
        operation_key: str,
    ) -> tuple[str | None, list[_ItemObservationState]]:
        """Latest authoritative state per provider item, derived from full history.

        Ordering is by the provider observation time (the authoritative order), with
        the append sequence as a deterministic tie-break, so a late-ingested older
        readback cannot silently overwrite newer evidence.
        """
        rows = conn.execute(
            """SELECT r.reconciliation_id,r.outcome,r.reason_codes_json,
                      o.observation_id,o.payout_item_id,o.status,o.observed_at,
                      o.raw_sha256,o.recipient,o.amount_cents
               FROM reconciliations r
               JOIN provider_observations o ON o.observation_id=r.observation_id
               WHERE r.operation_key=?""",
            (operation_key,),
        ).fetchall()
        states = [
            _ItemObservationState(
                payout_item_id=row["payout_item_id"],
                observation_id=row["observation_id"],
                reconciliation_id=row["reconciliation_id"],
                status=row["status"],
                outcome=row["outcome"],
                reason_codes=tuple(json.loads(row["reason_codes_json"])),
                observed_at=datetime.fromisoformat(row["observed_at"]),
                raw_sha256=row["raw_sha256"],
                recipient=row["recipient"],
                amount_cents=row["amount_cents"],
            )
            for row in rows
        ]
        if not states:
            return None, []
        states.sort(key=lambda state: (state.observed_at, state.reconciliation_id))
        latest: dict[str, _ItemObservationState] = {}
        for state in states:
            latest[state.payout_item_id] = state
        items = [latest[payout_item_id] for payout_item_id in sorted(latest)]
        outcomes = [item.outcome for item in items]
        if "mismatch" in outcomes:
            return "mismatch", items
        if "pending" in outcomes:
            return "pending", items
        if "uncertain" in outcomes:
            return "uncertain", items
        if all(outcome == "matched_completed" for outcome in outcomes):
            return "matched_completed", items
        return "failed", items

    def reconciliation_count(self, operation_key: str) -> int:
        with self._connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM reconciliations WHERE operation_key=?",
                (operation_key,),
            ).fetchone()[0]

    def reconciliation_outcome(self, operation_key: str) -> str:
        with self._connection() as conn:
            outcome, _ = self._latest_item_states(conn, operation_key)
            if outcome is None:
                raise PersistenceConflict("RECONCILIATION_NOT_FOUND")
            return outcome

    def reconciliation_history(self, operation_key: str) -> list[dict[str, object]]:
        """Every retained observation/reconciliation pair, in authoritative order."""
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT r.reconciliation_id,r.outcome,r.reason_codes_json,r.reconciled_at,
                          o.observation_id,o.payout_item_id,o.payout_batch_id,o.status,
                          o.recipient,o.currency,o.amount_cents,o.observed_at,o.raw_sha256
                   FROM reconciliations r
                   JOIN provider_observations o ON o.observation_id=r.observation_id
                   WHERE r.operation_key=?""",
                (operation_key,),
            ).fetchall()
        entries: list[tuple[tuple[datetime, int], dict[str, object]]] = []
        for row in rows:
            observed_at = datetime.fromisoformat(row["observed_at"])
            entries.append(
                (
                    (observed_at, row["reconciliation_id"]),
                    {
                        "reconciliation_id": row["reconciliation_id"],
                        "observation_id": row["observation_id"],
                        "payout_item_id": row["payout_item_id"],
                        "payout_batch_id": row["payout_batch_id"],
                        "provider_status": row["status"],
                        "outcome": row["outcome"],
                        "reason_codes": list(json.loads(row["reason_codes_json"])),
                        "recipient": row["recipient"],
                        "currency": row["currency"],
                        "amount_cents": row["amount_cents"],
                        "observed_at": row["observed_at"],
                        "raw_sha256": row["raw_sha256"],
                        "reconciled_at": row["reconciled_at"],
                    },
                )
            )
        entries.sort(key=lambda entry: entry[0])
        return [entry[1] for entry in entries]

    def latest_financial_outcome(self, operation_key: str) -> dict[str, object]:
        """The current financial reading of a payout, independent of phase state."""
        with self._connection() as conn:
            operation = conn.execute(
                """SELECT state,payout_batch_id,payout_item_id
                   FROM payment_operations WHERE operation_key=?""",
                (operation_key,),
            ).fetchone()
            if operation is None:
                raise PersistenceConflict("OPERATION_NOT_FOUND")
            outcome, items = self._latest_item_states(conn, operation_key)
            observation_count = conn.execute(
                "SELECT COUNT(*) FROM provider_observations WHERE operation_key=?",
                (operation_key,),
            ).fetchone()[0]
            reconciliation_count = conn.execute(
                "SELECT COUNT(*) FROM reconciliations WHERE operation_key=?",
                (operation_key,),
            ).fetchone()[0]
        latest = (
            max(items, key=lambda item: (item.observed_at, item.reconciliation_id))
            if items
            else None
        )
        return {
            "operation_key": operation_key,
            "operation_state": operation["state"],
            "outcome": outcome,
            "provider_status": latest.status if latest else None,
            "reason_codes": list(latest.reason_codes) if latest else [],
            "payout_item_id": latest.payout_item_id if latest else operation["payout_item_id"],
            "payout_batch_id": operation["payout_batch_id"],
            "observation_id": latest.observation_id if latest else None,
            "observed_at": latest.observed_at.isoformat() if latest else None,
            "observation_count": observation_count,
            "reconciliation_count": reconciliation_count,
        }

    def operation_binding(self, operation_key: str) -> dict[str, object]:
        with self._connection() as conn:
            row = conn.execute(
                """SELECT state,payout_batch_id,payout_item_id
                   FROM payment_operations WHERE operation_key=?""",
                (operation_key,),
            ).fetchone()
            if row is None:
                raise PersistenceConflict("OPERATION_NOT_FOUND")
            return dict(row)

    def record_observation_conflict(self, operation_key: str, detail: str) -> None:
        """Record a readback anomaly for an acknowledged payout without inventing evidence.

        Used when a provider response cannot be turned into a typed observation. It
        never fabricates a financial row and never builds a payout request.
        """
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                changed = conn.execute(
                    """UPDATE payment_operations
                       SET state='CONFLICT',detail=?,updated_at=?
                       WHERE operation_key=?
                         AND state IN ('ACKNOWLEDGED','VERIFIED','CONFLICT')""",
                    (detail, datetime.now(timezone.utc).isoformat(), operation_key),
                )
                if changed.rowcount != 1:
                    raise PersistenceConflict("OBSERVABLE_OPERATION_REQUIRED")
                self._append_event(
                    conn,
                    operation_key,
                    "PAYOUT_OBSERVATION_CONFLICT",
                    {"detail": detail},
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def recover_interrupted_dispatches(self) -> int:
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute(
                    "SELECT operation_key FROM payment_operations WHERE state='SENDING'"
                ).fetchall()
                now = datetime.now(timezone.utc).isoformat()
                for row in rows:
                    conn.execute(
                        """UPDATE payment_operations
                           SET state='UNCERTAIN',detail='PROCESS_INTERRUPTED_AFTER_DISPATCH_CLAIM',updated_at=?
                           WHERE operation_key=? AND state='SENDING'""",
                        (now, row["operation_key"]),
                    )
                    self._append_event(
                        conn,
                        row["operation_key"],
                        "PAYOUT_DISPATCH_UNCERTAIN",
                        {"detail": "PROCESS_INTERRUPTED_AFTER_DISPATCH_CLAIM"},
                    )
                conn.execute("COMMIT")
                return len(rows)
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def operation_state(self, operation_key: str) -> str:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT state FROM payment_operations WHERE operation_key=?",
                (operation_key,),
            ).fetchone()
            if row is None:
                raise PersistenceConflict("OPERATION_NOT_FOUND")
            return row["state"]

    @staticmethod
    def _intent_from_row(row: sqlite3.Row) -> PaymentIntent:
        return PaymentIntent(
            operation_key=row["operation_key"],
            claim_id=row["claim_id"],
            claim_revision=row["claim_revision"],
            evidence_sha256=row["evidence_sha256"],
            policy_version=row["policy_version"],
            fund_id=row["fund_id"],
            recipient=row["recipient"],
            amount=Money(row["currency"], row["amount_cents"]),
            authorization_id=row["authorization_id"],
        )

    @staticmethod
    def _claim_from_row(row: sqlite3.Row) -> Claim:
        return Claim(
            claim_id=row["claim_id"],
            volunteer_id=row["volunteer_id"],
            activity=Activity(
                row["activity_id"], row["activity_code"], date.fromisoformat(row["occurred_on"])
            ),
            evidence=EvidenceIdentity(
                row["evidence_sha256"],
                row["media_type"] or "application/octet-stream",
                row["original_name"] or "stored-evidence",
            ),
            requested=Money(row["currency"], row["amount_cents"]),
            recipient=row["recipient"],
            expense_category=row["expense_category"],
        )

    def audit_trail(self, aggregate_id: str) -> list[dict[str, object]]:
        """Append-only audit events for one aggregate, in sequence order.

        Returns payload metadata only. Payloads in this system never contain
        credential material, but the trail is rendered as structured fields rather
        than re-emitted as raw JSON so a rendering bug cannot echo one.
        """
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT event_type,payload_json,payload_sha256,occurred_at
                   FROM audit_events WHERE aggregate_id=? ORDER BY sequence""",
                (aggregate_id,),
            ).fetchall()
        return [
            {
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "payload_sha256": row["payload_sha256"],
                "occurred_at": row["occurred_at"],
            }
            for row in rows
        ]

    @staticmethod
    def _append_event(
        conn: sqlite3.Connection,
        aggregate_id: str,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        conn.execute(
            """INSERT INTO audit_events(
                event_id,aggregate_id,event_type,payload_json,payload_sha256,occurred_at
            ) VALUES(?,?,?,?,?,?)""",
            (
                f"event-{uuid4().hex}",
                aggregate_id,
                event_type,
                payload_json,
                sha256(payload_json.encode("utf-8")).hexdigest(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
