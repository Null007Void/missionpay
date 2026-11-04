"""Build the public release tree from an explicit inclusion manifest.

Publication is a gate. This script only ever writes to a local output directory; it
does not create a remote, push, or change visibility.

The public set is *included* by name, not filtered out of the full tree. An allow-list
fails safe: a newly added internal document is private until someone deliberately adds
it here, whereas a deny-list would leak it by default.

Usage:
    python scripts/release/build_public_tree.py --output ../missionpay-public
    python scripts/release/build_public_tree.py --check      # audit only, no writes
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import re
import shutil
import sys
import zipfile

REPO = Path(__file__).resolve().parents[2]

#: Files published as-is.
PUBLIC_FILES = (
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "README.md",
    "INSTALL.md",
    "ARCHITECTURE.md",
    "LIMITATIONS.md",
    "THIRD_PARTY.md",
    "DEMO.md",
    "pyproject.toml",
    "requirements-dev.txt",
)

#: Directory trees published in full.
PUBLIC_TREES = ("src", "tests", "scripts")

#: Never published, kept for the record so the exclusion is deliberate.
INTERNAL = (
    "PROJECT_STATUS.md",
    "COMPLETION-MAP.md",
    "MP-EVIDENCE-REPORT.md",
    "MP-LIFECYCLE-REPORT.md",
    "MP-RELEASE-008.md",
    "MP-RELEASE-009.md",
    "MP-RELEASE-010.md",
    "MP-RELEASE-011.md",
    "MP-RELEASE-012.md",
    "MP-RELEASE-013.md",
    "MP-RELEASE-014.md",
    "MP-RELEASE-015.md",
    "MP-RELEASE-016.md",
    "MP-RELEASE-017.md",
    "MP-RELEASE-018.md",
    "MP-RELEASE-019.md",
    "MP-RELEASE-020.md",
    "missionpay-qualification-report.md",
    "submission-audit.md",
    "submission-prep.md",
    "SUBMISSION_DRAFT_TEXT.md",
    "demo-evidence.md",
    "demo-script.md",
    "evidence/",
    "releases/",
)

#: Credential and personal-data shapes that must never reach a public tree.
SECRET_PATTERNS = (
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("slack_token", re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}")),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("paypal_secret_assignment", re.compile(r"PAYPAL_CLIENT_SECRET\s*=\s*[\"']?[A-Za-z0-9_-]{12,}")),
    ("bearer_literal", re.compile(r"Bearer\s+[A-Za-z0-9._-]{24,}")),
)

#: Absolute developer paths and workforce identifiers that must not ship.
PRIVATE_MARKERS = (
    "C:\\Users\\",
    "C:/Users/",
    "MissionPay-Private",
    "BOUNDARY Forge",
    "LivePatch",
    "CivicTrace",
    "BoundarySmith",
    "SwapSure",
)

#: The scanner necessarily contains the marker strings it searches for, so it cannot
#: scan itself. Exempting one named file is preferable to weakening every pattern.
SCAN_EXEMPT = frozenset({"scripts/release/build_public_tree.py"})

#: Documented synthetic fixtures allowed to look like credentials.
ALLOWED_LOOKALIKES = (
    "stub-token-not-a-credential",
    "synthetic-sandbox-secret-9d2f",
    "sandbox-client-secret-fixture-do-not-log-4f9c1a",
    "volunteer-demo-passphrase",
    "reviewer-demo-passphrase",
)

BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".whl", ".pyc"}

#: Generated or environment-local paths that must never be copied. The builder walks
#: the filesystem, so it does not inherit .gitignore and has to exclude these itself:
#: an editable install leaves src/*.egg-info behind and it is not source.
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "build",
    "dist",
    "venv",
    ".git",
}
EXCLUDED_SUFFIXES = (".egg-info", ".pyc", ".pyo", ".whl", ".sqlite3", ".db")


def _is_excluded(path: Path) -> bool:
    if path.parts[:2] == ("scripts", "replay"):
        return True
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return True
    if any(part.endswith(".egg-info") for part in path.parts):
        return True
    name = path.name
    if any(name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return True
    # SQLite WAL/SHM sidecars and runtime evidence directories.
    if name.endswith(("-wal", "-shm")) or name.endswith(".evidence"):
        return True
    return False


def _iter_public(repo: Path):
    for name in PUBLIC_FILES:
        path = repo / name
        if path.exists():
            yield path
    for tree in PUBLIC_TREES:
        root = repo / tree
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(repo)
            if _is_excluded(relative):
                continue
            yield path


def _skipped_generated(repo: Path) -> list[str]:
    """Generated/environment paths present in the tree but deliberately not published.

    Reported explicitly so the payload count can be reconciled against what is on
    disk: a reviewer should never have to guess whether ``src/missionpay.egg-info``
    was excluded on purpose or missed by accident.
    """
    skipped: list[str] = []
    for tree in PUBLIC_TREES:
        root = repo / tree
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and _is_excluded(path.relative_to(repo)):
                skipped.append(path.relative_to(repo).as_posix())
    return skipped


def scan_text(rel: str, text: str) -> list[str]:
    findings: list[str] = []
    for label, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0)
            if any(allowed in value for allowed in ALLOWED_LOOKALIKES):
                continue
            findings.append(f"{rel}: {label} -> {value[:34]}â€¦")
    for marker in PRIVATE_MARKERS:
        if marker in text:
            findings.append(f"{rel}: private marker -> {marker}")
    return findings


def audit(repo: Path) -> tuple[list[Path], list[str]]:
    files = list(_iter_public(repo))
    findings: list[str] = []
    for path in files:
        rel = path.relative_to(repo).as_posix()
        if rel in SCAN_EXEMPT:
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        findings.extend(scan_text(rel, text))
    return files, findings


def _write_archive(destination: Path, archive_path: Path, prefix: str) -> list[str]:
    """Zip the *exported payload*, never the private checkout.

    Entries are added in sorted order with a fixed timestamp so the archive is
    reproducible, and the member list is derived from the files actually written to
    the export directory rather than from the inclusion list. That is what makes
    membership verifiable: the archive cannot contain something the export did not.
    """
    members = sorted(p for p in destination.rglob("*") if p.is_file())
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in members:
            relative = path.relative_to(destination).as_posix()
            info = zipfile.ZipInfo(f"{prefix}/{relative}", date_time=(2026, 10, 31, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, path.read_bytes())
    return [f"{prefix}/{p.relative_to(destination).as_posix()}" for p in members]


def _verify_archive(archive_path: Path, expected: list[str]) -> list[str]:
    """Return a list of problems; empty means the archive matches the payload."""
    problems: list[str] = []
    with zipfile.ZipFile(archive_path) as bundle:
        names = bundle.namelist()
        if sorted(names) != sorted(expected):
            extra = sorted(set(names) - set(expected))
            missing = sorted(set(expected) - set(names))
            problems.append(f"membership mismatch: extra={extra[:5]} missing={missing[:5]}")
        for name in names:
            relative = name.split("/", 1)[1] if "/" in name else name
            if relative in SCAN_EXEMPT:
                continue
            # Internal entries are matched at the payload root, so a module named
            # ``evidence.py`` is not mistaken for the private ``evidence/`` directory.
            for internal in INTERNAL:
                bare = internal.rstrip("/")
                if relative == bare or relative.startswith(bare + "/"):
                    problems.append(f"{name}: internal document {bare}")
            payload = bundle.read(name)
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError:
                continue
            for marker in PRIVATE_MARKERS:
                if marker in text:
                    problems.append(f"{name}: private marker {marker}")
    return problems


MANIFEST_NAME = "MANIFEST.sha256"

#: Directories a build creates at the payload root. Root-only on purpose: a directory
#: named ``build`` deeper in the tree could legitimately be source, and silently
#: ignoring it would exclude real payload from the manifest check.
GENERATED_ROOT_DIRS = ("build", "dist")

#: Directory names that are never source, wherever they appear.
GENERATED_DIR_NAMES = ("__pycache__", ".pytest_cache", ".venv")

#: Directory suffixes identifying setuptools metadata. A suffix rather than a name
#: because the directory is named after the distribution: ``missionpay.egg-info``.
GENERATED_DIR_SUFFIXES = (".egg-info",)

#: A POSIX path with nothing that could resolve outside the tree on any platform.
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


def _is_generated(
    rel: str,
    *,
    root_dirs: tuple[str, ...] = (),
    dir_names: tuple[str, ...] = (),
    dir_suffixes: tuple[str, ...] = (),
) -> bool:
    """True when ``rel`` is build output rather than payload.

    Only *directory* components are considered, so a payload file whose name merely
    ends in ``.egg-info`` is still checked.
    """
    parts = rel.split("/")
    if parts and parts[0] in root_dirs:
        return True
    for part in parts[:-1]:
        if part in dir_names:
            return True
        if any(part.endswith(suffix) for suffix in dir_suffixes):
            return True
    return False


def manifest_bytes(entries) -> bytes:
    """An LF-terminated checksum manifest, sorted by path, never listing itself.

    Written as bytes rather than through :meth:`Path.write_text`, which translates
    newlines on Windows. A CR byte in a checksum file is not cosmetic: `sha256sum -c
    MANIFEST.sha256` on Linux or macOS then looks for a file literally named
    ``README.md\\r`` and fails for every entry, so a judge cannot verify the payload
    they were handed. The payload is portable; the manifest has to be too.
    """
    lines = sorted(f"{digest}  {path}" for path, digest in entries)
    return "".join(line + "\n" for line in lines).encode("utf-8")


def _path_problem(path: str) -> str | None:
    """Why a manifest path is unsafe, or ``None`` when it is a relative POSIX path.

    Runs before any filesystem access: the point is that a hostile or malformed
    manifest can never make the verifier open something outside the tree. A path is
    accepted only if it is relative, uses ``/``, and has no component that could
    escape or be reinterpreted by the platform.
    """
    if not path:
        return "empty"
    if "\\" in path:
        return "backslash"
    if path.startswith("/"):
        return "absolute"
    if _DRIVE_PREFIX.match(path):
        return "drive"  # C:/outside.txt and the drive-relative C:outside.txt alike
    if ":" in path:
        return "colon"  # drive-relative remainder or an NTFS alternate data stream
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        return "control"
    for part in path.split("/"):
        if part in ("", ".", ".."):
            return "traversal"
        if part != part.strip():
            return "whitespace"  # Windows silently strips it
        if part.endswith("."):
            return "trailing_dot"  # Windows silently strips it too
    return None


def parse_manifest(data: bytes) -> list[tuple[str, str]]:
    """Parse ``<sha256>  <relative/posix/path>`` lines, refusing anything unsafe."""
    if b"\r" in data:
        raise ValueError("MANIFEST_CONTAINS_CR")
    entries: list[tuple[str, str]] = []
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        digest, separator, path = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"MANIFEST_LINE_INVALID:{line[:40]}")
        problem = _path_problem(path)
        if problem is not None:
            raise ValueError(f"MANIFEST_PATH_REJECTED:{problem}")
        entries.append((path, digest))
    return entries


def payload_crlf(destination: Path) -> list[str]:
    """Text payload files carrying CRLF, which a POSIX checkout would not have."""
    offenders: list[str] = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        data = path.read_bytes()
        if b"\0" in data[:8000]:
            continue
        if b"\r\n" in data:
            offenders.append(path.relative_to(destination).as_posix())
    return offenders


def verify_tree(
    destination: Path,
    ignore_root_dirs: tuple[str, ...] = (),
    ignore_dir_names: tuple[str, ...] = (),
    ignore_dir_suffixes: tuple[str, ...] = (),
) -> list[str]:
    """Problems with a published tree; an empty list means it verifies.

    Checks both directions, because each hides a different mistake: a listed file
    that is absent means the payload is incomplete, and a present file that is not
    listed means something reached the tree without being accounted for.

    The ignore arguments exist for the build-from-extracted-source case: a checkout of
    the published archive is pristine until it is built in, after which
    ``build/`` and ``src/missionpay.egg-info/`` hold output the manifest never
    described. Publication passes none of them, so the public check stays strict.
    """
    problems: list[str] = []
    manifest = destination / MANIFEST_NAME
    if not manifest.is_file():
        return [f"{MANIFEST_NAME}: missing"]
    try:
        entries = parse_manifest(manifest.read_bytes())
    except ValueError as exc:
        return [f"{MANIFEST_NAME}: {exc}"]

    listed: set[str] = set()
    for path, digest in entries:
        if path == MANIFEST_NAME:
            problems.append(f"{MANIFEST_NAME}: lists itself")
            continue
        if path in listed:
            problems.append(f"{path}: listed twice")
        listed.add(path)
        target = destination / path
        if not target.is_file():
            problems.append(f"{path}: listed but missing")
            continue
        if sha256(target.read_bytes()).hexdigest() != digest:
            problems.append(f"{path}: digest differs from the manifest")

    for path in sorted(destination.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(destination).as_posix()
        if rel == MANIFEST_NAME:
            continue
        if _is_generated(
            rel,
            root_dirs=ignore_root_dirs,
            dir_names=ignore_dir_names,
            dir_suffixes=ignore_dir_suffixes,
        ):
            continue
        if rel not in listed:
            problems.append(f"{rel}: present but not listed")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="destination directory")
    parser.add_argument("--archive", default=None, help="write a sanitized source .zip here")
    parser.add_argument("--prefix", default="missionpay-1.0.0", help="archive member prefix")
    parser.add_argument("--check", action="store_true", help="audit without writing")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify an existing --output tree against its manifest; writes nothing",
    )
    args = parser.parse_args()

    if args.verify:
        if not args.output:
            print("--output is required with --verify", file=sys.stderr)
            return 2
        target = Path(args.output).resolve()
        problems = verify_tree(target)
        if problems:
            print(f"MANIFEST_VERIFY_FAILED={len(problems)}", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        entries = parse_manifest((target / MANIFEST_NAME).read_bytes())
        print(f"manifest_entries={len(entries)}")
        print("manifest_verified=payload_present_and_listed")
        return 0

    files, findings = audit(REPO)
    total_bytes = sum(path.stat().st_size for path in files)
    skipped = _skipped_generated(REPO)

    print(f"public_files={len(files)}")
    print(f"public_bytes={total_bytes}")
    print(f"generated_excluded={len(skipped)}")
    for name in skipped:
        print(f"  generated: {name}")
    print("excluded_internal:")
    for name in INTERNAL:
        print(f"  - {name}")

    if findings:
        print(f"\nSCAN_FINDINGS={len(findings)}", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        print("REFUSING_TO_BUILD: resolve the findings above", file=sys.stderr)
        return 1

    print("secret_scan=clean")

    if args.check:
        print("CHECK_ONLY: nothing written")
        return 0

    if not args.output:
        print("--output is required unless --check is used", file=sys.stderr)
        return 2

    destination = Path(args.output).resolve()
    if destination == REPO or REPO in destination.parents:
        print("refusing to write the public tree inside the private checkout", file=sys.stderr)
        return 2
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    entries: list[tuple[str, str]] = []
    for source in files:
        rel = source.relative_to(REPO)
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entries.append((rel.as_posix(), sha256(target.read_bytes()).hexdigest()))

    offenders = payload_crlf(destination)
    if offenders:
        print(f"REFUSING_TO_PUBLISH: {len(offenders)} payload file(s) carry CRLF", file=sys.stderr)
        for name in offenders:
            print(f"  {name}", file=sys.stderr)
        print("Normalize the working tree (git add --renormalize .) and rebuild.", file=sys.stderr)
        return 1

    (destination / MANIFEST_NAME).write_bytes(manifest_bytes(entries))

    print(f"written_to={destination}")
    print(f"manifest_entries={len(entries)}")

    manifest_problems = verify_tree(destination)
    if manifest_problems:
        print(f"MANIFEST_VERIFY_FAILED={len(manifest_problems)}", file=sys.stderr)
        for problem in manifest_problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("manifest_verified=payload_present_and_listed")

    if args.archive:
        archive_path = Path(args.archive).resolve()
        if REPO == archive_path or REPO in archive_path.parents:
            print("refusing to write the public archive inside the private checkout", file=sys.stderr)
            return 2
        members = _write_archive(destination, archive_path, args.prefix)
        print(f"archive={archive_path}")
        print(f"archive_entries={len(members)}")
        problems = _verify_archive(archive_path, members)
        if problems:
            print(f"ARCHIVE_VERIFY_FAILED={len(problems)}", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print("archive_verified=membership+hygiene")
        print(f"archive_sha256={sha256(archive_path.read_bytes()).hexdigest()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
