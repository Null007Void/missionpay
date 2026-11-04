"""Release-policy regressions: canonical line endings and portable checksums.

These tests exist because one commit once produced three different wheel digests
(`8db713ca…`, `98972a92…`, `f98b3839…`) depending on the checkout's line endings, and
because the published `MANIFEST.sha256` carried CR bytes that made `sha256sum -c` fail
on Linux and macOS.

They are git-free on purpose: the same suite runs from the public source tree, which
has no `.git`, so nothing here may shell out to git.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder_release():
    return _load("_mp_build_release", "scripts/release/build_release.py")


@pytest.fixture(scope="module")
def builder_public():
    return _load("_mp_build_public_tree", "scripts/release/build_public_tree.py")


# -- canonical line endings -------------------------------------------------


def test_gitattributes_pins_lf_for_every_checkout():
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in text
    assert "*.png binary" in text
    assert b"\r" not in (ROOT / ".gitattributes").read_bytes()


def test_packaged_sources_carry_no_crlf_on_disk(builder_release):
    """A CRLF working tree is what produced a wheel that was not the commit."""
    tracked = [
        path.relative_to(ROOT).as_posix()
        for directory in builder_release.PACKAGED_DIRS
        for path in sorted((ROOT / directory).rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc")
    ]
    tracked += [name for name in builder_release.PACKAGED_FILES if (ROOT / name).is_file()]
    offenders = [
        name
        for name in tracked
        if builder_release.is_text_path(name) and b"\r\n" in (ROOT / name).read_bytes()
    ]
    assert offenders == []


def test_packaged_inputs_are_the_package_and_its_metadata(builder_release):
    tracked = [
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "src/missionpay/web.py",
        "src/missionpay/py.typed",
        "tests/test_web_app.py",
        "scripts/release/build_release.py",
        "MP-RELEASE-008.md",
    ]
    assert builder_release.packaged_paths(tracked) == [
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "src/missionpay/py.typed",
        "src/missionpay/web.py",
    ]


# -- source divergence classification ---------------------------------------


def test_line_ending_divergence_is_named_as_a_line_ending_problem(builder_release):
    committed = {"src/missionpay/store.py": b"a\nb\nc\n"}
    disk = {"src/missionpay/store.py": b"a\r\nb\r\nc\r\n"}
    problems = builder_release.source_mismatches(
        ["src/missionpay/store.py"], disk.get, committed.get
    )
    assert problems == [("src/missionpay/store.py", "line_endings_differ_from_commit")]


def test_a_real_content_change_is_not_mistaken_for_line_endings(builder_release):
    committed = {"src/missionpay/store.py": b"amount = 1\n"}
    disk = {"src/missionpay/store.py": b"amount = 2\n"}
    problems = builder_release.source_mismatches(
        ["src/missionpay/store.py"], disk.get, committed.get
    )
    assert problems == [("src/missionpay/store.py", "bytes_differ_from_commit")]


def test_missing_files_are_reported_from_each_side(builder_release):
    problems = builder_release.source_mismatches(
        ["src/a.py", "src/b.py"],
        {"src/a.py": b"x\n"}.get,
        {"src/b.py": b"y\n"}.get,
    )
    assert problems == [
        ("src/a.py", "missing_from_commit"),
        ("src/b.py", "missing_from_worktree"),
    ]


def test_a_crlf_blob_is_refused_even_when_the_checkout_agrees(builder_release):
    committed = {"src/missionpay/store.py": b"a\r\nb\r\n"}
    problems = builder_release.source_mismatches(
        ["src/missionpay/store.py"], committed.get, committed.get
    )
    assert problems == [("src/missionpay/store.py", "committed_with_crlf")]


def test_a_matching_canonical_tree_is_accepted(builder_release):
    payload = b"a\nb\n"
    problems = builder_release.source_mismatches(
        ["src/missionpay/store.py"],
        read_disk=lambda _path: payload,
        read_committed=lambda _path: payload,
    )
    assert problems == []


# -- wheel member verification ---------------------------------------------


def test_wheel_members_must_match_the_commit(builder_release, tmp_path):
    wheel = tmp_path / "missionpay-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as bundle:
        bundle.writestr("missionpay/store.py", b"canonical\n")
        bundle.writestr("missionpay/py.typed", b"")
        bundle.writestr("missionpay-1.0.0.dist-info/RECORD", b"generated,,\n")
    good = lambda path: {
        "src/missionpay/store.py": b"canonical\n",
        "src/missionpay/py.typed": b"",
    }[path]
    assert builder_release.wheel_mismatches(wheel, good) == []

    crlf = lambda path: {
        "src/missionpay/store.py": b"canonical\n",
        "src/missionpay/py.typed": b"\r\n",
    }[path]
    assert builder_release.wheel_mismatches(wheel, crlf) == [
        "missionpay/py.typed: packaged bytes differ from src/missionpay/py.typed"
    ]


# -- portable checksum manifest --------------------------------------------


def test_manifest_is_lf_sorted_and_does_not_list_itself(builder_public):
    data = builder_public.manifest_bytes(
        [("README.md", "b" * 64), ("src/missionpay/web.py", "a" * 64)]
    )
    assert b"\r" not in data
    assert data.endswith(b"\n")
    lines = data.decode().splitlines()
    assert lines == sorted(lines), "a stable order is what makes two builds comparable"
    assert "MANIFEST.sha256" not in data.decode()


def test_manifest_paths_are_relative_posix(builder_public):
    data = builder_public.manifest_bytes([("src/missionpay/web.py", "a" * 64)])
    assert builder_public.parse_manifest(data) == [("src/missionpay/web.py", "a" * 64)]


@pytest.mark.parametrize(
    "path",
    ["README.md", "src/missionpay/web.py", "tests/test_web_app.py", "docs/a-b_c.1.md"],
)
def test_safe_manifest_paths_are_accepted(builder_public, path):
    assert builder_public.parse_manifest((("a" * 64) + "  " + path + "\n").encode()) == [(path, "a" * 64)]


@pytest.mark.parametrize(
    "path,reason",
    [
        ("nested/../../outside.txt", "traversal"),
        ("../outside.txt", "traversal"),
        ("src/./web.py", "traversal"),
        ("src//web.py", "traversal"),
        ("/etc/passwd", "absolute"),
        ("\\server\\share\\x", "backslash"),
        ("src\\missionpay\\web.py", "backslash"),
        ("C:/outside.txt", "drive"),
        ("C:outside.txt", "drive"),
        ("c:relative.txt", "drive"),
        ("src/web.py:stream", "colon"),
        ("README.md ", "whitespace"),
        ("README.md.", "trailing_dot"),
        ("src/\x01web.py", "control"),
        ("", "empty"),
    ],
)
def test_unsafe_manifest_paths_are_refused_before_any_lookup(builder_public, path, reason):
    with pytest.raises(ValueError) as excinfo:
        builder_public.parse_manifest((("a" * 64) + "  " + path + "\n").encode())
    assert f"MANIFEST_PATH_REJECTED:{reason}" in str(excinfo.value)


def test_verify_tree_stops_at_a_rejected_path_without_looking_outside(builder_public, tmp_path):
    """A manifest naming an escape is refused while parsing, before any read.

    A decoy file is planted one level above the tree: if verification resolved the
    path, it would find (or miss) something there rather than reporting a parse error.
    """
    _published_tree(tmp_path, builder_public)
    (tmp_path.parent / "outside.txt").write_bytes(b"decoy\n")
    (tmp_path / "MANIFEST.sha256").write_bytes(
        (("a" * 64) + "  nested/../../outside.txt\n").encode()
    )
    problems = builder_public.verify_tree(tmp_path)
    assert problems == ["MANIFEST.sha256: MANIFEST_PATH_REJECTED:traversal"]


def test_a_crlf_manifest_is_refused_rather_than_silently_accepted(builder_public):
    data = ("a" * 64 + "  README.md\r\n").encode()
    with pytest.raises(ValueError, match="MANIFEST_CONTAINS_CR"):
        builder_public.parse_manifest(data)


def _published_tree(root: Path, builder_public):
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_bytes(b"readme\n")
    (root / "src" / "web.py").write_bytes(b"code\n")
    entries = [
        ("README.md", __import__("hashlib").sha256(b"readme\n").hexdigest()),
        ("src/web.py", __import__("hashlib").sha256(b"code\n").hexdigest()),
    ]
    (root / "MANIFEST.sha256").write_bytes(builder_public.manifest_bytes(entries))


def test_a_verified_tree_reports_no_problems(builder_public, tmp_path):
    _published_tree(tmp_path, builder_public)
    assert builder_public.verify_tree(tmp_path) == []


def test_a_missing_payload_file_is_detected(builder_public, tmp_path):
    _published_tree(tmp_path, builder_public)
    (tmp_path / "src" / "web.py").unlink()
    assert builder_public.verify_tree(tmp_path) == ["src/web.py: listed but missing"]


def test_an_unexpected_file_is_detected(builder_public, tmp_path):
    _published_tree(tmp_path, builder_public)
    (tmp_path / "leftover.sqlite3-wal").write_bytes(b"x")
    assert builder_public.verify_tree(tmp_path) == [
        "leftover.sqlite3-wal: present but not listed"
    ]


def test_a_modified_payload_file_is_detected(builder_public, tmp_path):
    _published_tree(tmp_path, builder_public)
    (tmp_path / "README.md").write_bytes(b"readme edited\n")
    assert builder_public.verify_tree(tmp_path) == [
        "README.md: digest differs from the manifest"
    ]


def test_crlf_in_the_payload_is_detected_but_binaries_are_exempt(builder_public, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "web.py").write_bytes(b"a\r\nb\r\n")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    assert builder_public.payload_crlf(tmp_path) == ["src/web.py"]


def test_build_output_does_not_fail_the_payload_check(builder_public, tmp_path):
    """Building from an extracted archive generates a directory the manifest never had."""
    _published_tree(tmp_path, builder_public)
    (tmp_path / "build" / "lib").mkdir(parents=True)
    (tmp_path / "build" / "lib" / "generated.py").write_bytes(b"x\n")
    assert builder_public.verify_tree(tmp_path) == [
        "build/lib/generated.py: present but not listed"
    ]
    assert builder_public.verify_tree(
        tmp_path, ignore_root_dirs=("build",)
    ) == []


def test_strict_check_is_the_default(builder_public):
    """Publication must never inherit the relaxed arguments by accident."""
    import inspect

    signature = inspect.signature(builder_public.verify_tree)
    for name in ("ignore_root_dirs", "ignore_dir_names", "ignore_dir_suffixes"):
        assert signature.parameters[name].default == ()


def test_generated_bytecode_nested_in_the_tree_is_ignored(builder_public, tmp_path):
    """Importing the verifier inside an extracted payload writes bytecode into it."""
    _published_tree(tmp_path, builder_public)
    cache = tmp_path / "scripts" / "release" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "build_public_tree.cpython-311.pyc").write_bytes(b"\x00compiled")
    assert builder_public.verify_tree(tmp_path) == [
        "scripts/release/__pycache__/build_public_tree.cpython-311.pyc: present but not listed"
    ]
    assert builder_public.verify_tree(
        tmp_path, ignore_dir_names=("__pycache__",)
    ) == []


def test_setuptools_metadata_directory_is_ignored(builder_public, tmp_path):
    """The reported defect: `pip wheel` leaves src/missionpay.egg-info behind.

    It is named after the distribution, so a rule that looks for a directory called
    ".egg-info" misses it and a second build refuses its own predecessor's output.
    """
    _published_tree(tmp_path, builder_public)
    metadata = tmp_path / "src" / "missionpay.egg-info"
    metadata.mkdir(parents=True)
    for name in ("PKG-INFO", "SOURCES.txt", "requires.txt", "top_level.txt"):
        (metadata / name).write_bytes(b"generated\n")
    assert len(builder_public.verify_tree(tmp_path)) == 4
    assert builder_public.verify_tree(
        tmp_path, ignore_dir_suffixes=(".egg-info",)
    ) == []


def test_a_directory_named_build_inside_the_payload_is_still_checked(builder_public, tmp_path):
    """Ignoring generated paths must not become a blanket exclusion of real source."""
    _published_tree(tmp_path, builder_public)
    (tmp_path / "src" / "build").mkdir()
    (tmp_path / "src" / "build" / "packaging_helpers.py").write_bytes(b"real source\n")
    assert builder_public.verify_tree(
        tmp_path, ignore_root_dirs=("build",)
    ) == ["src/build/packaging_helpers.py: present but not listed"]


def test_repeat_builds_as_separate_processes_accept_the_same_extracted_tree(tmp_path):
    """Two independent invocations against one extracted tree, the second after a build.

    This is the reviewer's scenario: the first run leaves ``src/missionpay.egg-info``
    behind, and the second run must not call its predecessor's output a leak. Run as
    separate processes so a process-scoped cache cannot mask the failure.
    """
    import os
    import subprocess
    import sys

    root = tmp_path / "missionpay-1.0.0"
    (root / "src" / "missionpay").mkdir(parents=True)
    (root / "scripts" / "release").mkdir(parents=True)
    (root / "src" / "missionpay" / "__init__.py").write_bytes(b'"""fixture package."""\n')
    (root / "src" / "missionpay" / "py.typed").write_bytes(b"")
    (root / "README.md").write_bytes(b"# fixture\n")
    (root / "LICENSE").write_bytes(b"MIT\n")
    (root / "pyproject.toml").write_bytes(b'[project]\nname = "missionpay"\n')
    for script in ("build_release.py", "build_public_tree.py"):
        source = ROOT / "scripts" / "release" / script
        (root / "scripts" / "release" / script).write_bytes(source.read_bytes())

    # Manifest over the payload, as the publisher writes it.
    spec = importlib.util.spec_from_file_location("_fixture_pub", root / "scripts" / "release" / "build_public_tree.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.dont_write_bytecode, previous = True, sys.dont_write_bytecode
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    entries = [
        (path.relative_to(root).as_posix(), __import__("hashlib").sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    (root / "MANIFEST.sha256").write_bytes(module.manifest_bytes(entries))

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    def run():
        return subprocess.run(
            [sys.executable, "scripts/release/build_release.py", "--check-source"],
            cwd=root, capture_output=True, text=True, env=env,
        )

    first = run()
    assert first.returncode == 0, first.stderr
    assert "source_verified=the_published_manifest" in first.stdout

    metadata = root / "src" / "missionpay.egg-info"
    metadata.mkdir()
    for name in ("PKG-INFO", "SOURCES.txt", "top_level.txt"):
        (metadata / name).write_bytes(b"generated by the first build\n")

    second = run()
    assert second.returncode == 0, second.stderr
    assert "REFUSING" not in second.stderr

    # The same tree must still reject a genuine stray file: the ignore list is narrow.
    (root / "src" / "missionpay" / "leftover.py").write_bytes(b"x\n")
    third = run()
    assert third.returncode == 2
    assert "src/missionpay/leftover.py: present but not listed" in third.stderr
