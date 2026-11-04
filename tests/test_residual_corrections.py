"""Regressions for the three residual corrections.

1. The interpretations rebuild is atomic and repairs a database left interrupted by the
   earlier non-atomic version, keeping every interpretation and approval binding.
2. Reuse of an interpretation is keyed on the configuration about to run, so changing
   mode or model produces a new result instead of redirecting to the old one.
3. One ordering rule decides "current accepted interpretation" for the page, the
   approval, the intent and the dispatch, so the reviewer approves what they saw.
"""
from datetime import date
import re

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient

import missionpay.web as web_module
from missionpay.auth import AuthorizationDenied, Role
from missionpay.config import AiMode, ProviderMode, RuntimeConfig
from missionpay.domain import Activity, Claim, EvidenceIdentity, FundPolicy, Money
from missionpay.executor import PayoutExecutor
from missionpay.policy import evaluate_claim
from missionpay.store import MissionPayStore, PersistenceConflict
from missionpay.web import create_app

# The interpretations table exactly as it stood at 2c329e6, before the attempt column.
HISTORICAL_INTERPRETATIONS_DDL = """
CREATE TABLE interpretations (
    interpretation_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    claim_revision INTEGER NOT NULL,
    evidence_sha256 TEXT NOT NULL,
    ai_mode TEXT NOT NULL,
    model_id TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ACCEPTED','REFUSED')),
    error_code TEXT,
    output_sha256 TEXT NOT NULL,
    facts_json TEXT NOT NULL DEFAULT '[]',
    missing_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    UNIQUE (claim_id, claim_revision, evidence_sha256, ai_mode, model_id, config_sha256),
    FOREIGN KEY (claim_id, claim_revision) REFERENCES claims(claim_id, revision)
);
"""

HISTORICAL_COLUMNS = (
    "interpretation_id,claim_id,claim_revision,evidence_sha256,ai_mode,model_id,"
    "config_sha256,status,error_code,output_sha256,facts_json,missing_json,created_at"
)


def _store(tmp_path, name="m.sqlite3"):
    store = MissionPayStore(tmp_path / name, password_hasher=PasswordHasher(time_cost=1))
    store.initialize()
    return store


def _seed_people_and_fund(store):
    store.register_user("v1", "v@example.test", Role.VOLUNTEER, "pass-volunteer-1")
    store.register_user("r1", "r@example.test", Role.REVIEWER, "pass-reviewer-1")
    store.create_fund(FundPolicy(
        fund_id="RIVER_CLEANUP_FUND", version="2026.1",
        allowed_activity_codes=frozenset({"RIVER_CLEANUP"}),
        allowed_expense_categories=frozenset({"PROTECTIVE_SUPPLIES"}),
        starts_on=date(2026, 1, 1), ends_on=date(2026, 12, 31),
        available=Money("USD", 50000), receipt_required=True))


def _claim(store, claim_id="c1", digest="a" * 64):
    volunteer = store.login("v@example.test", "pass-volunteer-1")
    claim = Claim(
        claim_id=claim_id, volunteer_id="v1",
        activity=Activity("a1", "RIVER_CLEANUP", date(2026, 10, 18)),
        evidence=EvidenceIdentity(digest, "text/plain", "receipt.txt"),
        requested=Money("USD", 7899), recipient="sb@business.example.com",
        expense_category="PROTECTIVE_SUPPLIES")
    store.submit_claim(volunteer.token, claim, revision=1)
    store.record_policy_decision(
        evaluate_claim(claim, store.load_fund_policy("RIVER_CLEANUP_FUND", "2026.1"),
                       prior_evidence_digests=frozenset()), revision=1)
    return claim


def _downgrade_interpretations_to_historical_shape(store):
    """Rewrite the table to the pre-attempt schema, preserving every row."""
    import sqlite3

    with sqlite3.connect(store.path) as conn:
        rows = conn.execute(
            f"SELECT {HISTORICAL_COLUMNS} FROM interpretations"
        ).fetchall()
        conn.execute("DROP TABLE interpretations")
        conn.executescript(HISTORICAL_INTERPRETATIONS_DDL)
        conn.executemany(
            f"INSERT INTO interpretations({HISTORICAL_COLUMNS}) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows)
        conn.commit()
    return len(rows)


# ---------------------------------------------------------------- 1. atomic migration

def test_migration_preserves_records_and_bindings(tmp_path):
    store = _store(tmp_path)
    _seed_people_and_fund(store)
    claim = _claim(store)
    accepted = store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
        ai_mode="fixture", model_id="fixture/demo", config_sha256="1" * 64,
        status="ACCEPTED", output_sha256="2" * 64)
    refused = store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
        ai_mode="fixture", model_id="fixture/other", config_sha256="3" * 64,
        status="REFUSED", error_code="CITATION_MISMATCH", output_sha256="4" * 64)
    reviewer = store.login("r@example.test", "pass-reviewer-1")
    authorization = store.authorize_claim(
        reviewer.token, "c1", revision=1,
        interpretation_id=accepted["interpretation_id"],
        interpretation_output_sha256=accepted["output_sha256"])
    intent = store.create_payment_intent(authorization.authorization_id)

    assert _downgrade_interpretations_to_historical_shape(store) == 2

    reopened = MissionPayStore(store.path, password_hasher=PasswordHasher(time_cost=1))
    reopened.initialize()

    history = reopened.interpretation_history("c1", 1)
    assert len(history) == 2
    assert {row["interpretation_id"] for row in history} == {
        accepted["interpretation_id"], refused["interpretation_id"]}
    assert all(row["attempt"] == 1 for row in history)

    binding = reopened.authorization_binding(intent.operation_key)
    assert binding["interpretation_id"] == accepted["interpretation_id"]
    assert binding["interpretation_output_sha256"] == accepted["output_sha256"]

    reloaded = reopened.load_intent(intent.operation_key)
    assert reloaded.recipient == "sb@business.example.com"
    assert reopened.authorization_binding(intent.operation_key)[
        "interpretation_id"] == accepted["interpretation_id"]


@pytest.mark.parametrize("boundary", [
    "_create_migrated_interpretations",
    "_fill_migrated_interpretations",
    "_drop_unmigrated_interpretations",
    "_rename_migrated_interpretations",
])
def test_migration_is_atomic_at_every_destructive_boundary(tmp_path, monkeypatch, boundary):
    store = _store(tmp_path)
    _seed_people_and_fund(store)
    claim = _claim(store)
    store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
        ai_mode="fixture", model_id="fixture/demo", config_sha256="1" * 64,
        status="ACCEPTED", output_sha256="2" * 64)
    assert _downgrade_interpretations_to_historical_shape(store) == 1

    def boom(*_args, **_kwargs):
        raise RuntimeError(f"injected failure at {boundary}")

    monkeypatch.setattr(MissionPayStore, boundary, staticmethod(boom))
    broken = MissionPayStore(store.path, password_hasher=PasswordHasher(time_cost=1))
    with pytest.raises(RuntimeError, match="injected failure"):
        broken.initialize()
    monkeypatch.undo()

    # The interrupted attempt must leave no partial table and no lost rows.
    import sqlite3

    with sqlite3.connect(store.path) as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        rows = conn.execute("SELECT COUNT(*) FROM interpretations").fetchone()[0]
        columns = {row[1] for row in conn.execute("PRAGMA table_info(interpretations)")}
    assert "interpretations_migrated" not in tables
    assert rows == 1
    assert "attempt" not in columns, "the failed transaction must roll back entirely"

    # And a clean restart completes the migration without losing the row.
    reopened = MissionPayStore(store.path, password_hasher=PasswordHasher(time_cost=1))
    reopened.initialize()
    history = reopened.interpretation_history("c1", 1)
    assert len(history) == 1
    assert history[0]["attempt"] == 1


def test_interrupted_state_from_the_old_migration_is_repaired(tmp_path):
    """A copy orphaned by the old non-atomic migration is promoted, not abandoned."""
    import sqlite3

    store = _store(tmp_path)
    _seed_people_and_fund(store)
    claim = _claim(store)
    store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
        ai_mode="fixture", model_id="fixture/demo", config_sha256="1" * 64,
        status="ACCEPTED", output_sha256="2" * 64)
    _downgrade_interpretations_to_historical_shape(store)

    # Reproduce the exact wreckage: copy made, old table dropped, rename never ran.
    with sqlite3.connect(store.path) as conn:
        conn.execute("""CREATE TABLE interpretations_migrated (
            interpretation_id TEXT PRIMARY KEY, claim_id TEXT NOT NULL,
            claim_revision INTEGER NOT NULL, evidence_sha256 TEXT NOT NULL,
            ai_mode TEXT NOT NULL, model_id TEXT NOT NULL, config_sha256 TEXT NOT NULL,
            attempt INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL, error_code TEXT,
            output_sha256 TEXT NOT NULL, facts_json TEXT NOT NULL DEFAULT '[]',
            missing_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL)""")
        conn.execute(
            f"INSERT INTO interpretations_migrated({HISTORICAL_COLUMNS},attempt) "
            f"SELECT {HISTORICAL_COLUMNS},1 FROM interpretations")
        conn.execute("DROP TABLE interpretations")
        conn.commit()

    reopened = MissionPayStore(store.path, password_hasher=PasswordHasher(time_cost=1))
    reopened.initialize()

    with sqlite3.connect(store.path) as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        rows = conn.execute("SELECT COUNT(*) FROM interpretations").fetchone()[0]
    assert "interpretations_migrated" not in tables
    assert rows == 1
    assert reopened.accepted_interpretation("c1", 1) is not None


# ---------------------------------------------------------------- 2. configuration reuse

def _demo_app(tmp_path, config):
    from missionpay.bootstrap import bootstrap_demo_store

    store = MissionPayStore(tmp_path / "ui.sqlite3", password_hasher=PasswordHasher(time_cost=1))
    store.initialize()
    bootstrap_demo_store(store.path)
    client = TestClient(create_app(store, config=config), base_url="http://testserver")
    return store, client


def _csrf(client, path="/app"):
    return re.search(r'name="csrf_token" value="([^"]+)"',
                     client.get(path).text).group(1)


def _sign_in_and_submit(client):
    client.post("/login", data={
        "email": "volunteer@missionpay.local",
        "password": "volunteer-demo-passphrase",
        "csrf_token": _csrf(client, "/login")}, follow_redirects=False)
    submitted = client.post("/claims/demo", data={"csrf_token": _csrf(client)},
                            follow_redirects=False)
    return submitted.headers["location"]


def _ollama_config(tmp_path, model="qwen2.5:3b"):
    return RuntimeConfig(
        provider_mode=ProviderMode.MOCK, ai_mode=AiMode.OLLAMA, ollama_model=model,
        database_path=str(tmp_path / "ui.sqlite3"), secure_cookies=False)


def test_reuse_requires_the_same_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(web_module, "OllamaJsonModel",
                        lambda **_kw: web_module._DemoInterpretationModel())
    config = _ollama_config(tmp_path)
    store, client = _demo_app(tmp_path, config)
    path = _sign_in_and_submit(client)

    calls = {"n": 0}
    real = web_module._interpret

    def counting(config_, evidence, text):
        calls["n"] += 1
        return real(config_, evidence, text)

    monkeypatch.setattr(web_module, "_interpret", counting)

    client.post(f"{path}/interpret", data={"csrf_token": _csrf(client)})
    first = calls["n"]
    assert first >= 1, "a request for Ollama must not be satisfied by the fixture result"

    # Same configuration: reuse, no second inference.
    client.post(f"{path}/interpret", data={"csrf_token": _csrf(client)})
    assert calls["n"] == first

    # A different model is a different configuration and must run again.
    (tmp_path / "other").mkdir(parents=True, exist_ok=True)
    other_store, other_client = _demo_app(tmp_path / "other", _ollama_config(
        tmp_path / "other", model="qwen3:4b"))
    other_path = _sign_in_and_submit(other_client)
    monkeypatch.setattr(web_module, "_interpret", counting)
    before = calls["n"]
    other_client.post(f"{other_path}/interpret", data={"csrf_token": _csrf(other_client)})
    assert calls["n"] == before + 1


def test_fixture_result_is_not_served_to_a_different_mode(tmp_path, monkeypatch):
    """The reproduced defect: an accepted fixture row must not short-circuit Ollama."""
    monkeypatch.setattr(web_module, "OllamaJsonModel",
                        lambda **_kw: web_module._DemoInterpretationModel())
    store, client = _demo_app(tmp_path, _ollama_config(tmp_path))
    path = _sign_in_and_submit(client)

    token = store.login("volunteer@missionpay.local", "volunteer-demo-passphrase").token
    claim = store.claim_snapshot(token, "river-safety-001")["claim"]
    store.record_interpretation(
        claim_id="river-safety-001", revision=1, evidence_sha256=claim.evidence.sha256,
        ai_mode="fixture", model_id="fixture/demo-interpretation-v1",
        config_sha256="f" * 64, status="ACCEPTED", output_sha256="1" * 64)

    calls = {"n": 0}
    real = web_module._interpret

    def counting(config_, evidence, text):
        calls["n"] += 1
        return real(config_, evidence, text)

    monkeypatch.setattr(web_module, "_interpret", counting)
    client.post(f"{path}/interpret", data={"csrf_token": _csrf(client)})

    assert calls["n"] == 1
    history = store.interpretation_history("river-safety-001", 1)
    assert len(history) == 2
    assert {row["ai_mode"] for row in history} == {"fixture", "ollama"}


# ---------------------------------------------------------------- 3. one ordering rule

def _accept(store, model, config, output, when):
    import sqlite3

    record = store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256="a" * 64, ai_mode="fixture",
        model_id=model, config_sha256=config, status="ACCEPTED", output_sha256=output)
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE interpretations SET created_at=? WHERE interpretation_id=?",
                     (when, record["interpretation_id"]))
        conn.commit()
    return record


def test_attempt_numbers_do_not_rank_across_configurations(tmp_path):
    """A succeeds on attempt 2 but is older than B's attempt 1: the page and the
    authority must both choose B, and approving B must be accepted."""
    store = _store(tmp_path)
    _seed_people_and_fund(store)
    _claim(store)

    store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256="a" * 64, ai_mode="fixture",
        model_id="cfg-A", config_sha256="a" * 64, status="REFUSED", output_sha256="9" * 64)
    _accept(store, "cfg-A", "a" * 64, "1" * 64, "2026-10-31T09:00:00+00:00")
    newer = _accept(store, "cfg-B", "b" * 64, "2" * 64, "2026-10-31T12:00:00+00:00")

    page_choice = store.accepted_interpretation("c1", 1)
    assert page_choice["interpretation_id"] == newer["interpretation_id"]

    reviewer = store.login("r@example.test", "pass-reviewer-1")
    authorization = store.authorize_claim(
        reviewer.token, "c1", revision=1,
        interpretation_id=page_choice["interpretation_id"],
        interpretation_output_sha256=page_choice["output_sha256"])
    intent = store.create_payment_intent(authorization.authorization_id)
    assert store.authorization_binding(intent.operation_key)[
        "interpretation_id"] == newer["interpretation_id"]
    assert intent.amount.cents == 7899
    assert store.authorization_binding(intent.operation_key)[
        "interpretation_id"] == newer["interpretation_id"]


def test_a_newer_result_makes_the_older_page_stale(tmp_path):
    store = _store(tmp_path)
    _seed_people_and_fund(store)
    _claim(store)

    older = _accept(store, "cfg-A", "a" * 64, "1" * 64, "2026-10-31T09:00:00+00:00")
    _accept(store, "cfg-B", "b" * 64, "2" * 64, "2026-10-31T12:00:00+00:00")

    reviewer = store.login("r@example.test", "pass-reviewer-1")
    with pytest.raises(AuthorizationDenied, match="INTERPRETATION_STALE"):
        store.authorize_claim(
            reviewer.token, "c1", revision=1,
            interpretation_id=older["interpretation_id"],
            interpretation_output_sha256=older["output_sha256"])


def test_ordering_tie_breaks_deterministically(tmp_path):
    """Identical timestamps fall back to attempt, then id, at every call site."""
    store = _store(tmp_path)
    _seed_people_and_fund(store)
    _claim(store)

    refused = store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256="a" * 64, ai_mode="fixture",
        model_id="cfg-A", config_sha256="a" * 64, status="REFUSED", output_sha256="9" * 64)
    with __import__("sqlite3").connect(store.path) as conn:
        conn.execute("UPDATE interpretations SET created_at=? WHERE interpretation_id=?",
                     ("2026-10-31T09:00:00+00:00", refused["interpretation_id"]))
        conn.commit()
    second = _accept(store, "cfg-A", "a" * 64, "2" * 64, "2026-10-31T09:00:00+00:00")

    current = store.accepted_interpretation("c1", 1)
    assert current["interpretation_id"] == second["interpretation_id"]
    assert current["attempt"] == 2

    reviewer = store.login("r@example.test", "pass-reviewer-1")
    authorization = store.authorize_claim(
        reviewer.token, "c1", revision=1,
        interpretation_id=current["interpretation_id"],
        interpretation_output_sha256=current["output_sha256"])
    intent = store.create_payment_intent(authorization.authorization_id)
    assert store.authorization_binding(intent.operation_key)[
        "interpretation_id"] == second["interpretation_id"]
    assert refused["interpretation_id"] != second["interpretation_id"]


class _Adapter:
    """Synthetic PayPal transport: records calls, never touches a network."""

    def __init__(self):
        self.calls = 0

    def create_payout(self, intent, *, access_token):
        self.calls += 1
        return {"payout_batch_id": "BATCH-1", "batch_status": "PENDING"}


def test_dispatch_binds_the_same_row_the_page_displayed(tmp_path):
    """UI -> approval -> intent -> dispatch must all name one interpretation."""
    store = _store(tmp_path)
    _seed_people_and_fund(store)
    _claim(store)

    store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256="a" * 64, ai_mode="fixture",
        model_id="cfg-A", config_sha256="a" * 64, status="REFUSED", output_sha256="9" * 64)
    _accept(store, "cfg-A", "a" * 64, "1" * 64, "2026-10-31T09:00:00+00:00")
    newer = _accept(store, "cfg-B", "b" * 64, "2" * 64, "2026-10-31T12:00:00+00:00")

    displayed = store.accepted_interpretation("c1", 1)
    reviewer = store.login("r@example.test", "pass-reviewer-1")
    authorization = store.authorize_claim(
        reviewer.token, "c1", revision=1,
        interpretation_id=displayed["interpretation_id"],
        interpretation_output_sha256=displayed["output_sha256"])
    intent = store.create_payment_intent(authorization.authorization_id)

    adapter = _Adapter()
    executor = PayoutExecutor(store, adapter, max_payout_cents=10_000,
                              expected_recipient="sb@business.example.com")
    executor.execute(
        intent.operation_key,
        token_provider=lambda: "synthetic-token",
        owner_id="r1",
    )

    assert adapter.calls == 1
    bound = store.authorization_binding(intent.operation_key)
    assert bound["interpretation_id"] == displayed["interpretation_id"]
    assert bound["interpretation_id"] == newer["interpretation_id"]
    assert store.load_intent(intent.operation_key).recipient == "sb@business.example.com"


# ------------------------------------------- orphan recovery: prove before dropping

ORPHAN_DDL = """
CREATE TABLE interpretations_migrated (
    interpretation_id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL,
    claim_revision INTEGER NOT NULL,
    evidence_sha256 TEXT NOT NULL,
    ai_mode TEXT NOT NULL,
    model_id TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL,
    error_code TEXT,
    output_sha256 TEXT NOT NULL,
    facts_json TEXT NOT NULL DEFAULT '[]',
    missing_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
"""

_COLS = ("interpretation_id,claim_id,claim_revision,evidence_sha256,ai_mode,model_id,"
         "config_sha256,attempt,status,error_code,output_sha256,facts_json,missing_json,created_at")


def _live_store_with_rows(tmp_path):
    import sqlite3

    store = _store(tmp_path)
    _seed_people_and_fund(store)
    claim = _claim(store)
    store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
        ai_mode="fixture", model_id="fixture/demo", config_sha256="1" * 64,
        status="ACCEPTED", output_sha256="2" * 64)
    store.record_interpretation(
        claim_id="c1", revision=1, evidence_sha256=claim.evidence.sha256,
        ai_mode="fixture", model_id="fixture/other", config_sha256="3" * 64,
        status="REFUSED", error_code="CITATION_MISMATCH", output_sha256="4" * 64)
    with sqlite3.connect(store.path) as conn:
        conn.execute(_SCHEMA_ORPHAN := ORPHAN_DDL)
        conn.commit()
    return store


def _snapshot(db_path):
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        tables = sorted(row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"))
        out = {"tables": tables}
        for name in ("interpretations", "interpretations_migrated"):
            if name in tables:
                out[name] = sorted(tuple(row) for row in conn.execute(
                    f"SELECT {_COLS} FROM {name}"))
        return out


def _copy_live_into_orphan(store):
    import sqlite3

    with sqlite3.connect(store.path) as conn:
        conn.execute(f"INSERT INTO interpretations_migrated({_COLS}) "
                     f"SELECT {_COLS} FROM interpretations")
        conn.commit()


def test_orphan_that_exactly_duplicates_live_is_dropped(tmp_path):
    store = _live_store_with_rows(tmp_path)
    _copy_live_into_orphan(store)
    assert "interpretations_migrated" in _snapshot(store.path)["tables"]

    reopened = MissionPayStore(store.path, password_hasher=PasswordHasher(time_cost=1))
    reopened.initialize()

    snapshot = _snapshot(store.path)
    assert "interpretations_migrated" not in snapshot["tables"]
    assert len(snapshot["interpretations"]) == 2
    assert reopened.accepted_interpretation("c1", 1) is not None


def test_orphan_with_a_record_absent_from_live_refuses_and_keeps_both(tmp_path):
    import sqlite3

    store = _live_store_with_rows(tmp_path)
    _copy_live_into_orphan(store)
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            f"INSERT INTO interpretations_migrated({_COLS}) VALUES"
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("interp-orphan-only", "c1", 1, "a" * 64, "fixture", "fixture/orphan",
             "5" * 64, 1, "ACCEPTED", None, "6" * 64, "[]", "[]",
             "2026-10-31T13:00:00+00:00"))
        conn.commit()
    before = _snapshot(store.path)

    reopened = MissionPayStore(store.path, password_hasher=PasswordHasher(time_cost=1))
    with pytest.raises(PersistenceConflict) as excinfo:
        reopened.initialize()

    message = str(excinfo.value)
    assert "INTERPRETATION_MIGRATION_AMBIGUOUS" in message
    assert "orphan_records_absent_from_live=1" in message
    assert "kept_both_tables=true" in message
    assert _snapshot(store.path) == before, "refusal must leave both tables untouched"


def test_orphan_conflicting_on_the_same_id_refuses_and_keeps_both(tmp_path):
    import sqlite3

    store = _live_store_with_rows(tmp_path)
    _copy_live_into_orphan(store)
    with sqlite3.connect(store.path) as conn:
        # Same primary key, different monetary-relevant content.
        conn.execute("UPDATE interpretations_migrated SET output_sha256=? "
                     "WHERE model_id='fixture/demo'", ("7" * 64,))
        conn.commit()
    before = _snapshot(store.path)

    reopened = MissionPayStore(store.path, password_hasher=PasswordHasher(time_cost=1))
    with pytest.raises(PersistenceConflict) as excinfo:
        reopened.initialize()

    message = str(excinfo.value)
    assert "same_id_different_contents=1" in message
    assert "kept_both_tables=true" in message
    assert _snapshot(store.path) == before


def test_orphan_with_distinct_records_in_both_tables_refuses(tmp_path):
    import sqlite3

    store = _live_store_with_rows(tmp_path)
    # The orphan holds a different interpretation entirely; the live table holds two.
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            f"INSERT INTO interpretations_migrated({_COLS}) VALUES"
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("interp-distinct", "c1", 1, "a" * 64, "fixture", "fixture/distinct",
             "8" * 64, 1, "ACCEPTED", None, "9" * 64, "[]", "[]",
             "2026-10-31T14:00:00+00:00"))
        conn.commit()
    before = _snapshot(store.path)
    assert len(before["interpretations"]) == 2
    assert len(before["interpretations_migrated"]) == 1

    reopened = MissionPayStore(store.path, password_hasher=PasswordHasher(time_cost=1))
    with pytest.raises(PersistenceConflict):
        reopened.initialize()

    assert _snapshot(store.path) == before


def test_empty_orphan_beside_a_populated_live_table_is_dropped(tmp_path):
    store = _live_store_with_rows(tmp_path)  # orphan created empty
    reopened = MissionPayStore(store.path, password_hasher=PasswordHasher(time_cost=1))
    reopened.initialize()

    snapshot = _snapshot(store.path)
    assert "interpretations_migrated" not in snapshot["tables"]
    assert len(snapshot["interpretations"]) == 2


def test_missing_live_table_is_promoted_from_the_orphan(tmp_path):
    import sqlite3

    store = _live_store_with_rows(tmp_path)
    _copy_live_into_orphan(store)
    with sqlite3.connect(store.path) as conn:
        conn.execute("DROP TABLE interpretations")
        conn.commit()
    before = _snapshot(store.path)

    reopened = MissionPayStore(store.path, password_hasher=PasswordHasher(time_cost=1))
    reopened.initialize()

    snapshot = _snapshot(store.path)
    assert "interpretations_migrated" not in snapshot["tables"]
    assert snapshot["interpretations"] == before["interpretations_migrated"]
