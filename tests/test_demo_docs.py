"""Documentation integrity for the documented local demo path.

The README run instructions are a testable claim: the default demo database file, the safe
replay instructions, and the duplicate-submission outcome must stay true to the shipped code and must
not dirty a clean checkout.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEMO_DB = "missionpay-demo.sqlite3"


def test_documented_demo_names_its_state_file_and_safe_replay():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # The default database name is resolved in config.py (the single owner of
    # environment defaults); entrypoint.py only consumes the resolved value.
    config = (ROOT / "src" / "missionpay" / "config.py").read_text(encoding="utf-8")
    entrypoint = (ROOT / "src" / "missionpay" / "entrypoint.py").read_text(encoding="utf-8")

    assert DEMO_DB in config, "the default demo database name changed; update README"
    assert DEMO_DB not in entrypoint, "entrypoint must not keep a second hardcoded default"
    assert DEMO_DB in readme
    assert "new empty dedicated data directory" in readme
    assert "DUPLICATE_REIMBURSEMENT_BLOCKED" in readme
    assert "MISSIONPAY_DB" in readme


def test_default_demo_database_is_ignored_by_git():
    ignore_lines = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert DEMO_DB in ignore_lines, "documented demo run would leave an untracked database file"


def test_wal_sidecars_are_ignored_and_replay_preserves_them():
    """Replay uses a new directory and preserves existing SQLite sidecars."""
    ignore_lines = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert f"{DEMO_DB}-*" in ignore_lines, "WAL sidecars are not ignored"
    assert "WAL/SHM sidecars" in readme
    assert "rm -f" not in readme
