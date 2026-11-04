"""Evidence intake: limits, path handling, digest stability, and web wiring."""

from datetime import date
from hashlib import sha256
import io
import re

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient

from missionpay.auth import Role
from missionpay.domain import FundPolicy, Money
from missionpay.evidence import (
    MAX_EVIDENCE_BYTES,
    EvidenceRejected,
    EvidenceStore,
    sanitise_filename,
)
from missionpay.store import MissionPayStore
from missionpay.web import create_app

RECEIPT = b"Hardware Depot\nWork gloves\nTotal $42.10\n"


@pytest.fixture
def store(tmp_path):
    return EvidenceStore(tmp_path / "evidence")


# ------------------------------------------------------------- intake rules


def test_round_trip_is_content_addressed(store):
    saved = store.save(data=RECEIPT, media_type="text/plain", filename="receipt.txt")
    assert saved.sha256 == sha256(RECEIPT).hexdigest()
    assert saved.size_bytes == len(RECEIPT)
    assert saved.path.name == f"{saved.sha256}.txt"
    assert store.read_bytes(saved.sha256) == RECEIPT
    assert store.read_text(saved.sha256) == RECEIPT.decode()


def test_identical_content_is_stored_once(store):
    first = store.save(data=RECEIPT, media_type="text/plain", filename="a.txt")
    second = store.save(data=RECEIPT, media_type="text/plain", filename="b.txt")
    assert first.path == second.path
    assert len(list(store.root.iterdir())) == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\config", "config"),
        ("/absolute/path/receipt.txt", "receipt.txt"),
        ("receipt\x00.txt", "receipt.txt"),
        ("  spaced name .txt  ", "spaced name .txt"),
    ],
)
def test_filenames_reduce_to_a_basename(raw, expected):
    assert sanitise_filename(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "/", "..", "../..", "\x00"])
def test_unusable_filenames_are_refused(raw):
    with pytest.raises(EvidenceRejected):
        sanitise_filename(raw)


def test_traversal_filename_cannot_escape_the_store(store):
    saved = store.save(data=RECEIPT, media_type="text/plain", filename="../../escape.txt")
    assert saved.path.parent == store.root
    assert saved.original_name == "escape.txt"
    assert not (store.root.parent / "escape.txt").exists()


def test_oversize_body_is_refused_before_writing(store):
    oversized = b"x" * (MAX_EVIDENCE_BYTES + 1)
    with pytest.raises(EvidenceRejected, match="EVIDENCE_TOO_LARGE"):
        store.save(data=oversized, media_type="text/plain", filename="big.txt")
    assert not store.root.exists() or list(store.root.iterdir()) == []


def test_empty_body_is_refused(store):
    with pytest.raises(EvidenceRejected, match="EVIDENCE_EMPTY"):
        store.save(data=b"", media_type="text/plain", filename="empty.txt")


@pytest.mark.parametrize("media", ["application/pdf", "text/html", "application/octet-stream", ""])
def test_disallowed_media_types_are_refused(store, media):
    with pytest.raises(EvidenceRejected, match="EVIDENCE_TYPE_NOT_ALLOWED"):
        store.save(data=RECEIPT, media_type=media, filename="x.txt")


def test_media_type_parameters_are_tolerated(store):
    saved = store.save(data=RECEIPT, media_type="text/plain; charset=utf-8", filename="x.txt")
    assert saved.media_type == "text/plain"


def test_read_refuses_a_non_digest_key(store):
    for bad in ("../../etc/passwd", "A" * 64, "abc"):
        with pytest.raises(EvidenceRejected, match="EVIDENCE_DIGEST_INVALID"):
            store.read_bytes(bad)


def test_missing_digest_raises_not_found(store):
    with pytest.raises(EvidenceRejected, match="EVIDENCE_NOT_FOUND"):
        store.read_bytes("b" * 64)


def test_binary_evidence_has_no_text_layer(store):
    saved = store.save(data=b"\x89PNG\r\n\x1a\n\xff\xfe", media_type="image/png", filename="r.png")
    assert store.read_text(saved.sha256) is None
    assert store.read_bytes(saved.sha256).startswith(b"\x89PNG")


def test_default_root_never_collides_with_the_tracked_evidence_directory(tmp_path):
    """A default-database run must not write uploads into documentation.

    The repository tracks an ``evidence/`` directory of engineering records. If the
    runtime blob root were ``<db parent>/evidence``, a demo run at the repo root would
    drop uploads into version-controlled documentation.
    """
    from missionpay.evidence import default_evidence_root

    database = tmp_path / "missionpay-demo.sqlite3"
    root = default_evidence_root(database)

    assert root.name == "missionpay-demo.sqlite3.evidence"
    assert root.parent == database.parent
    assert root != database.parent / "evidence"


# ------------------------------------------------------------- web wiring


def _csrf(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match
    return match.group(1)


def _web_store(tmp_path) -> MissionPayStore:
    store = MissionPayStore(tmp_path / "up.sqlite3", password_hasher=PasswordHasher(time_cost=1))
    store.initialize()
    store.register_user("v1", "v@example.test", Role.VOLUNTEER, "volunteer-passphrase")
    store.register_user("r1", "r@example.test", Role.REVIEWER, "reviewer-passphrase")
    store.create_fund(
        FundPolicy(
            fund_id="RIVER_CLEANUP_FUND",
            version="2026.1",
            allowed_activity_codes=frozenset({"RIVER_CLEANUP"}),
            allowed_expense_categories=frozenset({"PROTECTIVE_SUPPLIES"}),
            starts_on=date(2026, 1, 1),
            ends_on=date(2026, 12, 31),
            available=Money("USD", 50_000),
            receipt_required=True,
        )
    )
    return store


def _login(client):
    page = client.get("/login")
    response = client.post(
        "/login",
        data={"email": "v@example.test", "password": "volunteer-passphrase", "csrf_token": _csrf(page.text)},
        follow_redirects=False,
    )
    assert response.status_code == 303


def test_upload_creates_a_policy_evaluated_claim(tmp_path):
    store = _web_store(tmp_path)
    client = TestClient(create_app(store), base_url="https://testserver")
    _login(client)

    response = client.post(
        "/claims/upload",
        data={
            "csrf_token": _csrf(client.get("/app").text),
            "amount_cents": "4210",
            "activity_code": "RIVER_CLEANUP",
            "expense_category": "PROTECTIVE_SUPPLIES",
        },
        files={"evidence": ("my-receipt.txt", io.BytesIO(RECEIPT), "text/plain")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    detail = client.get(response.headers["location"])
    assert detail.status_code == 200
    assert "$42.10" in detail.text
    assert sha256(RECEIPT).hexdigest() in detail.text
    assert "my-receipt.txt" in detail.text


def test_upload_refuses_a_disallowed_type_without_storing(tmp_path):
    store = _web_store(tmp_path)
    client = TestClient(create_app(store), base_url="https://testserver")
    _login(client)

    response = client.post(
        "/claims/upload",
        data={
            "csrf_token": _csrf(client.get("/app").text),
            "amount_cents": "4210",
            "activity_code": "RIVER_CLEANUP",
            "expense_category": "PROTECTIVE_SUPPLIES",
        },
        files={"evidence": ("payload.html", io.BytesIO(b"<script>"), "text/html")},
    )
    assert response.status_code == 400
    assert "EVIDENCE_TYPE_NOT_ALLOWED" in response.text


def test_upload_refuses_an_oversize_body(tmp_path):
    store = _web_store(tmp_path)
    client = TestClient(create_app(store), base_url="https://testserver")
    _login(client)

    response = client.post(
        "/claims/upload",
        data={
            "csrf_token": _csrf(client.get("/app").text),
            "amount_cents": "4210",
            "activity_code": "RIVER_CLEANUP",
            "expense_category": "PROTECTIVE_SUPPLIES",
        },
        files={
            "evidence": (
                "big.txt",
                io.BytesIO(b"x" * (MAX_EVIDENCE_BYTES + 10)),
                "text/plain",
            )
        },
    )
    assert response.status_code == 400
    assert "EVIDENCE_TOO_LARGE" in response.text


def test_upload_requires_volunteer_role(tmp_path):
    store = _web_store(tmp_path)
    client = TestClient(create_app(store), base_url="https://testserver")
    page = client.get("/login")
    client.post(
        "/login",
        data={"email": "r@example.test", "password": "reviewer-passphrase", "csrf_token": _csrf(page.text)},
        follow_redirects=False,
    )
    response = client.post(
        "/claims/upload",
        data={
            "csrf_token": _csrf(client.get("/app").text),
            "amount_cents": "4210",
            "activity_code": "RIVER_CLEANUP",
            "expense_category": "PROTECTIVE_SUPPLIES",
        },
        files={"evidence": ("r.txt", io.BytesIO(RECEIPT), "text/plain")},
    )
    assert response.status_code == 403
