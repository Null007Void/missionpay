"""Private credential entry for PayPal sandbox, designed so a secret never leaks.

Threat model for this module: the risk is not an attacker reading memory, it is a
secret ending up somewhere durable and readable — a shell history entry, a command
argument visible in a process list, a log line, a traceback, a screenshot, a committed
file. So credentials are read from a file the owner creates, never from an argument,
and nothing here ever returns a value to a caller that prints.

Rules enforced:

* the credential file must live **outside** the repository, so it cannot be committed
  by accident even if someone runs `git add -A`;
* the file is created by an explicit `init` step that refuses to overwrite;
* every reporting path returns presence booleans and file metadata, never values;
* parse errors name the line number and the key, never the content.

Environment variables still win over the file, because that is the normal CI and
container mechanism. Precedence is documented and tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import stat
from typing import Mapping

#: Overrides the default location. Useful when the owner keeps secrets on another volume.
CREDENTIALS_FILE_ENV = "MISSIONPAY_CREDENTIALS_FILE"

#: The keys a genuine sandbox run requires.
REQUIRED_KEYS = ("PAYPAL_CLIENT_ID", "PAYPAL_CLIENT_SECRET")

#: Optional keys that may also live in the file.
OPTIONAL_KEYS = (
    "MISSIONPAY_SANDBOX_RECIPIENT",
    "MISSIONPAY_MAX_PAYOUT_CENTS",
    "PAYPAL_BASE_URL",
    "MISSIONPAY_HOSTED_API_KEY",
    "MISSIONPAY_HOSTED_MODEL",
    "MISSIONPAY_HOSTED_BASE_URL",
)

#: Keys `status` reports presence for. PayPal's two remain the completeness criterion —
#: a hosted key is not "required" for a sandbox payout, and vice versa — but the owner
#: must be able to see whether the hosted key was found without printing a value.
REPORTED_KEYS = REQUIRED_KEYS + ("MISSIONPAY_HOSTED_API_KEY",)

_ALL_KEYS = REQUIRED_KEYS + OPTIONAL_KEYS

_TEMPLATE = """# MissionPay credentials — PayPal SANDBOX and hosted-model key.
#
# Fill in the two PayPal values from your PayPal developer app:
#   developer.paypal.com -> Apps & Credentials -> Sandbox -> <your app> -> Client ID / Secret
#
# Optional: a hosted model key, used only when MISSIONPAY_AI_MODE=hosted and
# MISSIONPAY_HOSTED_AI_CONFIRM=1 are both set. Leave it empty to keep the AI stage
# offline (fixture or loopback Ollama).
#
# This file must stay outside the MissionPay repository. Never commit it, paste it
# into chat, or pass its contents as a command argument.
#
# Nothing here is read until a genuine sandbox run or a hosted model call is started.

# Required for MISSIONPAY_PROVIDER_MODE=paypal_sandbox
PAYPAL_CLIENT_ID=
PAYPAL_CLIENT_SECRET=

# Required for MISSIONPAY_AI_MODE=hosted (environment wins over this file)
MISSIONPAY_HOSTED_API_KEY=
# MISSIONPAY_HOSTED_MODEL=gpt-4.1-mini

# Recommended: the sandbox account that receives the payout, and a hard cap in cents.
# MISSIONPAY_SANDBOX_RECIPIENT=sb-recipient@business.example.com
# MISSIONPAY_MAX_PAYOUT_CENTS=500
"""


class CredentialError(ValueError):
    """Credential handling refused. Messages never contain a value."""


def repo_root() -> Path:
    """Module-anchored checkout root, for the source-checkout case only.

    This is *not* a reliable way to find the user's repository: when MissionPay is
    installed as a wheel this points into site-packages. Repository exclusion is
    therefore decided primarily by :func:`_enclosing_work_tree`, which looks outward
    from the credential file itself.
    """
    return Path(__file__).resolve().parents[2]


def _enclosing_work_tree(path: Path) -> Path | None:
    """Walk up from *the file's own directory* looking for a repository marker.

    Works for an installed wheel, a source checkout, and a repository the package
    knows nothing about. ``.git`` is a directory in a normal clone and a file in a
    worktree or submodule, so existence is the right test.
    """
    try:
        start = path.expanduser().resolve().parent
    except OSError:  # pragma: no cover - defensive
        return None
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def default_credentials_path() -> Path:
    """A per-user location outside any repository."""
    override = os.environ.get(CREDENTIALS_FILE_ENV)
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "missionpay" / "credentials.env"
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "missionpay" / "credentials.env"


def _assert_outside_repository(path: Path) -> None:
    """Refuse any location a version-control system would track.

    Two independent checks, because either can be blind on its own:

    1. the file's own enclosing work tree — correct even when MissionPay is installed
       as a wheel and knows nothing about the caller's repository;
    2. the module-anchored source checkout — catches an exported tarball with no
       ``.git`` present.
    """
    try:
        resolved = path.expanduser().resolve()
    except OSError as exc:  # pragma: no cover - defensive
        raise CredentialError("CREDENTIALS_PATH_UNRESOLVABLE") from exc

    if _enclosing_work_tree(resolved) is not None:
        raise CredentialError("CREDENTIALS_FILE_MUST_BE_OUTSIDE_REPOSITORY")

    try:
        resolved.relative_to(repo_root())
    except ValueError:
        return
    raise CredentialError("CREDENTIALS_FILE_MUST_BE_OUTSIDE_REPOSITORY")


def write_template(path: Path | None = None) -> Path:
    """Create an empty credential file. Refuses to overwrite an existing one."""
    target = (path or default_credentials_path()).expanduser()
    _assert_outside_repository(target)
    if target.exists():
        raise CredentialError("CREDENTIALS_FILE_EXISTS")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_TEMPLATE, encoding="utf-8")
    _restrict_permissions(target)
    return target


def _restrict_permissions(path: Path) -> None:
    """Best effort: owner-only on POSIX. Windows ACLs are left to the user's profile."""
    if os.name == "nt":
        return
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:  # pragma: no cover - filesystem dependent
        pass


def parse(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines. Comments and blanks are ignored.

    Errors identify a *position*, never a content: echoing an unrecognised key name
    would publish whatever the user typed before the ``=``, which may itself be a
    secret pasted in the wrong place. Unknown keys are still rejected rather than
    ignored, because a typo would otherwise look like a missing credential.
    """
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise CredentialError(f"CREDENTIALS_LINE_MALFORMED:line={number}")
        key, _, value = line.partition("=")
        key = key.strip()
        if key not in _ALL_KEYS:
            # Deliberately no key text and no value text.
            raise CredentialError(f"CREDENTIALS_UNKNOWN_KEY:line={number}")
        cleaned = value.strip().strip('"').strip("'")
        if cleaned:
            values[key] = cleaned
    return values


def read_file(path: Path) -> dict[str, str]:
    _assert_outside_repository(path)
    try:
        return parse(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except UnicodeDecodeError as exc:
        raise CredentialError("CREDENTIALS_FILE_NOT_UTF8") from exc


def resolve(environ: Mapping[str, str], path: Path | None = None) -> dict[str, str]:
    """Merge credentials with environment taking precedence over the file."""
    merged = read_file(path or default_credentials_path())
    for key in _ALL_KEYS:
        from_env = environ.get(key)
        if from_env and from_env.strip():
            merged[key] = from_env.strip()
    return merged


@dataclass(frozen=True, slots=True)
class CredentialStatus:
    """Presence-only view, safe to print."""

    path: Path
    exists: bool
    readable: bool
    keys_present: tuple[str, ...] = field(default=())
    keys_missing: tuple[str, ...] = field(default=())
    permission_warning: str | None = None

    @property
    def complete(self) -> bool:
        return not self.keys_missing

    def describe(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "readable": self.readable,
            "complete": self.complete,
            "keys_present": list(self.keys_present),
            "keys_missing": list(self.keys_missing),
            "permission_warning": self.permission_warning,
        }


def status(path: Path | None = None) -> CredentialStatus:
    target = (path or default_credentials_path()).expanduser()
    exists = target.is_file()
    readable = False
    present: tuple[str, ...] = ()
    warning: str | None = None

    if exists:
        if os.name != "nt":
            mode = target.stat().st_mode
            if mode & (stat.S_IRGRP | stat.S_IROTH):
                warning = "CREDENTIALS_FILE_IS_GROUP_OR_WORLD_READABLE"
        try:
            values = read_file(target)
            readable = True
            present = tuple(k for k in REPORTED_KEYS if k in values)
        except CredentialError as exc:
            warning = str(exc)

    return CredentialStatus(
        path=target,
        exists=exists,
        readable=readable,
        keys_present=present,
        keys_missing=tuple(k for k in REQUIRED_KEYS if k not in present),
        permission_warning=warning,
    )
