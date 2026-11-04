"""Build the release artifact reproducibly from a clean, committed tree.

Why this exists: a wheel built by `pip wheel` embeds file modification times into the
zip container, so two builds from identical source produce identical *contents* but
different bytes. That makes a published SHA-256 unverifiable — a reviewer rebuilding
the same commit gets a different hash and cannot tell whether the source differs.
Pinning `SOURCE_DATE_EPOCH` to the commit timestamp makes the container deterministic.

Line endings are the second half of the same problem, and the one that actually bit
this project. A Windows checkout with `core.autocrlf=true` writes CRLF into the working
tree while git keeps LF in the index, so a wheel built from that tree packages bytes
that are *not* the committed bytes: one commit produced three digests (one per checkout
style) and only one of them matched the published artifact. `.gitattributes` now pins
`eol=lf`, and this script refuses to build when a packaged file on disk differs from its
committed blob — and refuses to accept a wheel whose members differ from the commit —
so "which bytes am I attesting to" is answered before the build, not after a reviewer
reports a hash mismatch.

This script also refuses to build from a dirty tree, for the same reason: an artifact
that does not correspond to a commit cannot be reviewed.

Usage:
    python scripts/release/build_release.py                     # build into dist/
    python scripts/release/build_release.py --out ../candidate  # build elsewhere
    python scripts/release/build_release.py --verify-only       # rebuild twice, compare
    python scripts/release/build_release.py --replace           # allow replacing a wheel
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import os
import shutil
import subprocess
import sys
import zipfile
import zlib

REPO = Path(__file__).resolve().parents[2]

#: Fixed release timestamp. Deliberately NOT derived from the commit time: a later
#: documentation-only commit would then change the epoch and therefore the artifact
#: hash, even though the packaged bytes are identical. A constant keeps the wheel
#: reproducible from any checkout whose packaged files match.
RELEASE_TIMESTAMP = "2026-10-31T00:00:00+00:00"

#: Everything the wheel is built from. `README.md` becomes the long description in
#: METADATA and `LICENSE` is copied into `dist-info/licenses/`, so both are build
#: inputs even though neither is a module.
PACKAGED_DIRS = ("src",)
PACKAGED_FILES = ("README.md", "LICENSE", "pyproject.toml")

#: Files copied into `dist-info/` verbatim, for the post-build byte check.
DIST_INFO_COPIES = {"missionpay-1.0.0.dist-info/licenses/LICENSE": "LICENSE"}

TEXT_SUFFIXES = (".py", ".md", ".toml", ".txt", ".yaml", ".yml", ".cfg", ".ini", ".json")

MANIFEST_NAME = "MANIFEST.sha256"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout


def _tree_state() -> tuple[bool, str, str]:
    dirty = _git("status", "--porcelain=v1", "--untracked-files=all")
    commit = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    return (dirty == ""), commit, tree


def _public_builder():
    """The published-tree verifier, loaded from the sibling script.

    Reusing one implementation is the point: a second manifest parser would let the
    producer and the consumer of ``MANIFEST.sha256`` disagree about what it means.

    Bytecode writing is suppressed for the import: this runs inside an extracted
    payload, and a ``__pycache__`` directory created here would be an unlisted file
    in the very tree being verified.
    """
    import importlib.util

    path = Path(__file__).with_name("build_public_tree.py")
    spec = importlib.util.spec_from_file_location("_mp_public_tree", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _resolve_source():
    """Resolve what the artifact must correspond to: a commit, or a published manifest.

    Returns ``(header_lines, packaged_paths, reference_reader, label, problems)``.
    The same wheel has to be reachable from a git checkout and from an extracted
    published archive, so the identity check follows whichever source is present.
    """
    if (REPO / ".git").exists():
        clean, commit, tree = _tree_state()
        if not clean:
            print("REFUSING_TO_BUILD: working tree is dirty", file=sys.stderr)
            print(_git("status", "--porcelain=v1", "--untracked-files=all"), file=sys.stderr)
            raise SystemExit(2)
        paths = packaged_paths(_git("ls-files").splitlines())
        reference = lambda p: _git_bytes("show", f"HEAD:{p}")  # noqa: E731
        problems = [
            f"{path}: {reason}"
            for path, reason in source_mismatches(
                paths,
                read_disk=lambda p: (REPO / p).read_bytes() if (REPO / p).exists() else None,
                read_committed=reference,
            )
        ]
        return [f"commit={commit}", f"tree={tree}", "source=git_commit"], paths, reference, "the commit", problems

    manifest = REPO / MANIFEST_NAME
    if not manifest.is_file():
        raise SystemExit(
            f"REFUSING_TO_BUILD: no .git and no {MANIFEST_NAME}, so the source identity is "
            "unknown; build from a git checkout or an extracted published archive"
        )
    builder = _public_builder()
    entries = builder.parse_manifest(manifest.read_bytes())
    paths = packaged_paths([path for path, _digest in entries])
    # Both directions: every listed file present and matching, and no unlisted file
    # present except output a build is allowed to generate. The generated-path policy
    # lives in one place — the module that also publishes — so the producer and this
    # consumer cannot drift apart.
    problems = builder.verify_tree(
        REPO,
        ignore_root_dirs=builder.GENERATED_ROOT_DIRS,
        ignore_dir_names=builder.GENERATED_DIR_NAMES,
        ignore_dir_suffixes=builder.GENERATED_DIR_SUFFIXES,
    )
    reference = lambda p: (REPO / p).read_bytes() if (REPO / p).exists() else None  # noqa: E731
    header = ["source=published_manifest", f"manifest_entries={len(entries)}"]
    return header, paths, reference, "the published manifest", problems


# -- canonical source checks ------------------------------------------------


def packaged_paths(tracked: list[str]) -> list[str]:
    """The tracked paths the artifact is built from, in a stable order."""
    selected = [
        path
        for path in tracked
        if path in PACKAGED_FILES
        or any(path == directory or path.startswith(directory + "/") for directory in PACKAGED_DIRS)
    ]
    return sorted(selected)


def is_text_path(path: str) -> bool:
    return path.endswith(TEXT_SUFFIXES)


def source_mismatches(paths, read_disk, read_committed):
    """Names and reasons for every packaged file that is not the committed bytes.

    Kept free of git so the policy itself can be tested with plain dictionaries:
    ``read_disk(path)`` and ``read_committed(path)`` each return bytes or ``None``.
    """
    problems: list[tuple[str, str]] = []
    for path in paths:
        committed = read_committed(path)
        if committed is None:
            problems.append((path, "missing_from_commit"))
            continue
        disk = read_disk(path)
        if disk is None:
            problems.append((path, "missing_from_worktree"))
            continue
        if disk != committed:
            if disk.replace(b"\r\n", b"\n") == committed.replace(b"\r\n", b"\n"):
                problems.append((path, "line_endings_differ_from_commit"))
            else:
                problems.append((path, "bytes_differ_from_commit"))
        elif is_text_path(path) and b"\r\n" in committed:
            # A committed CRLF blob is non-canonical even when the checkout agrees with it.
            problems.append((path, "committed_with_crlf"))
    return problems


def wheel_mismatches(wheel: Path, read_committed) -> list[str]:
    """Members whose bytes are not the source of record, e.g. a CRLF working tree."""
    problems: list[str] = []
    with zipfile.ZipFile(wheel) as bundle:
        for name in bundle.namelist():
            if name.startswith("missionpay/"):
                source = "src/" + name
            elif name in DIST_INFO_COPIES:
                source = DIST_INFO_COPIES[name]
            else:
                continue  # generated metadata: RECORD, METADATA, WHEEL, entry_points
            committed = read_committed(source)
            if committed is None:
                problems.append(f"{name}: no source {source}")
            elif bundle.read(name) != committed:
                problems.append(f"{name}: packaged bytes differ from {source}")
    return problems


def backend_from_wheel(wheel: Path) -> str:
    """The build backend version the wheel itself records."""
    with zipfile.ZipFile(wheel) as bundle:
        try:
            text = bundle.read("missionpay-1.0.0.dist-info/WHEEL").decode("utf-8")
        except KeyError:
            return "unknown"
    for line in text.splitlines():
        if line.lower().startswith("generator:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


# -- build ------------------------------------------------------------------


def _prepare_out(out: Path, replace: bool) -> None:
    if out != REPO / "dist" and (out == REPO or REPO in out.parents):
        raise SystemExit(
            "REFUSING_TO_BUILD: --out inside the checkout must be dist/, so a candidate "
            "build cannot overwrite source"
        )
    if out.exists():
        if not out.is_dir():
            raise SystemExit(f"REFUSING_TO_BUILD: {out} is not a directory")
        foreign = [p for p in out.iterdir() if p.suffix != ".whl"]
        if foreign:
            raise SystemExit(
                "REFUSING_TO_BUILD: output directory holds non-artifact files: "
                + ", ".join(sorted(p.name for p in foreign)[:5])
            )
        existing = sorted(out.glob("*.whl"))
        if existing and not replace:
            raise SystemExit(
                "REFUSING_TO_REPLACE_EXISTING_ARTIFACT: "
                + ", ".join(p.name for p in existing)
                + " (pass --replace to rebuild in place; preserve a superseded artifact by "
                "building to a new --out directory instead)"
            )
        for path in existing:
            path.unlink()
    else:
        out.mkdir(parents=True)


def _build_once(source_date_epoch: str, out: Path) -> Path:
    build = REPO / "build"
    if build.exists():
        shutil.rmtree(build)
    env = dict(os.environ, SOURCE_DATE_EPOCH=source_date_epoch)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(out)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-2000:], file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit("BUILD_FAILED")
    wheels = sorted(out.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel, found {len(wheels)}")
    return wheels[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true", help="rebuild twice and compare")
    parser.add_argument("--out", default=None, help="output directory (default: dist/)")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="allow replacing a wheel already present in the output directory",
    )
    parser.add_argument(
        "--check-source",
        action="store_true",
        help="verify the source identity and exit without building",
    )
    args = parser.parse_args()

    header, paths, reference, label, problems = _resolve_source()
    if problems:
        print(
            f"REFUSING_TO_BUILD: {len(problems)} packaged file(s) are not the source of record",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "The artifact would not correspond to this source. Check out with .gitattributes "
            "enforced (eol=lf) — or re-extract the published archive — and try again.",
            file=sys.stderr,
        )
        return 2

    for line in header:
        print(line)
    print(f"packaged_inputs={len(paths)}")
    if args.check_source:
        # A reviewer (and the repeat-build regression) can verify the tree's identity
        # without a build, so this path must not need pip, the backend or a network.
        print("source_verified=" + label.replace(" ", "_"))
        print("NO_ARTIFACT_BUILT")
        return 0

    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH") or str(
        int(datetime.fromisoformat(RELEASE_TIMESTAMP).timestamp())
    )
    out = Path(args.out).expanduser().resolve() if args.out else REPO / "dist"
    print(f"SOURCE_DATE_EPOCH={source_date_epoch}")
    # Recorded because it is a build input: the container bytes depend on the zip
    # implementation (POSIX mode vs Windows attributes) and the deflate library.
    print(f"platform={sys.platform}")
    print(f"python={sys.version.split()[0]}")
    print(f"zlib={zlib.ZLIB_RUNTIME_VERSION}")

    _prepare_out(out, args.replace)
    wheel = _build_once(source_date_epoch, out)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    print(f"out={out}")
    print(f"wheel={wheel.name}")
    print(f"size={wheel.stat().st_size}")
    print(f"sha256={digest}")
    print(f"backend={backend_from_wheel(wheel)}")

    mismatches = wheel_mismatches(wheel, reference)
    if mismatches:
        print(
            f"REFUSING_TO_ACCEPT: {len(mismatches)} packaged member(s) are not {label}",
            file=sys.stderr,
        )
        for problem in mismatches:
            print(f"  {problem}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(wheel) as bundle:
        print(f"wheel_members={len(bundle.namelist())}")
    print("member_check=" + label.replace(" ", "_"))

    if args.verify_only:
        second = _build_once(source_date_epoch, out)
        second_digest = hashlib.sha256(second.read_bytes()).hexdigest()
        print(f"rebuild_sha256={second_digest}")
        if second_digest != digest:
            print("NOT_REPRODUCIBLE: the two builds differ", file=sys.stderr)
            return 1
        print("REPRODUCIBLE: byte-identical across two builds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
