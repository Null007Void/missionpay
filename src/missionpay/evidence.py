"""Content-addressed evidence storage with hard intake limits.

Evidence arrives from an untrusted browser, so the stored name is derived only
from the content digest. The uploaded filename is never used to build a path; it
is sanitised, validated, and kept as metadata on the claim. That removes traversal
from the design rather than trying to filter it.

Intake limits are explicit constants, not defaults inherited from a framework:

* 256 KiB maximum body;
* three declared media types;
* a non-empty body;
* a filename that reduces to something non-empty after basename extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import inspect
from pathlib import Path
import os
import re

MAX_EVIDENCE_BYTES = 256 * 1024

#: Declared media type mapped to the extension used for the stored blob.
ALLOWED_MEDIA = {
    "text/plain": ".txt",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]")


class EvidenceRejected(ValueError):
    """Intake refused before anything is written."""


@dataclass(frozen=True, slots=True)
class StoredEvidence:
    sha256: str
    media_type: str
    original_name: str
    size_bytes: int
    path: Path


def sanitise_filename(raw: str) -> str:
    """Reduce an untrusted filename to a safe display name.

    ``../../etc/passwd`` becomes ``passwd``; a name that reduces to nothing is
    refused rather than replaced with a guess.
    """
    if not raw or not raw.strip():
        raise EvidenceRejected("EVIDENCE_FILENAME_REQUIRED")
    # Normalise both separators so a Windows-style path cannot survive on POSIX.
    base = raw.replace("\\", "/").rsplit("/", 1)[-1].strip()
    base = base.replace("\x00", "")
    if base in {"", ".", ".."}:
        raise EvidenceRejected("EVIDENCE_FILENAME_INVALID")
    cleaned = _SAFE_NAME.sub("_", base).strip(" .")
    if not cleaned:
        raise EvidenceRejected("EVIDENCE_FILENAME_INVALID")
    return cleaned[:120]


def default_evidence_root(database_path: str | Path) -> Path:
    """Where evidence blobs live for a given database file.

    Deliberately *not* a generic ``evidence/`` directory beside the database: this
    repository tracks an ``evidence/`` directory of engineering records, and a
    default-database demo run must never write uploads into documentation.
    """
    return Path(str(database_path) + ".evidence")


async def read_upload_bounded(upload: object, *, max_bytes: int = MAX_EVIDENCE_BYTES) -> bytes:
    """Read an upload without ever buffering more than ``max_bytes + 1``.

    ``UploadFile.read()`` with no argument buffers the entire body before any size
    check can run, so a hostile or accidental multi-gigabyte upload would be
    allocated first and rejected second. Asking for one byte more than the limit is
    enough to detect oversize while keeping the allocation bounded.

    Accepts a sync or async ``read`` so a plain test double works without a loop.
    """
    read = getattr(upload, "read", None)
    if read is None:
        raise EvidenceRejected("EVIDENCE_UPLOAD_UNREADABLE")
    data = read(max_bytes + 1)
    if inspect.isawaitable(data):
        data = await data
    if data is None:
        raise EvidenceRejected("EVIDENCE_UPLOAD_UNREADABLE")
    if len(data) > max_bytes:
        raise EvidenceRejected("EVIDENCE_TOO_LARGE")
    return data


class EvidenceStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, *, data: bytes, media_type: str, filename: str) -> StoredEvidence:
        if not data:
            raise EvidenceRejected("EVIDENCE_EMPTY")
        if len(data) > MAX_EVIDENCE_BYTES:
            raise EvidenceRejected("EVIDENCE_TOO_LARGE")
        normalised_type = (media_type or "").split(";")[0].strip().lower()
        extension = ALLOWED_MEDIA.get(normalised_type)
        if extension is None:
            raise EvidenceRejected("EVIDENCE_TYPE_NOT_ALLOWED")

        original_name = sanitise_filename(filename)
        digest = sha256(data).hexdigest()

        self.root.mkdir(parents=True, exist_ok=True)
        # Digest-derived name: the caller-supplied string cannot influence the path.
        target = self.root / f"{digest}{extension}"

        if target.exists():
            # Reuse is only safe if the existing bytes are actually what the name
            # claims. A truncated or overwritten blob would otherwise be silently
            # accepted for this digest and every future claim that cites it.
            if target.read_bytes() != data:
                raise EvidenceRejected("EVIDENCE_BLOB_MISMATCH")
        else:
            self._publish_atomically(target, data)

        return StoredEvidence(
            sha256=digest,
            media_type=normalised_type,
            original_name=original_name,
            size_bytes=len(data),
            path=target,
        )

    @staticmethod
    def _publish_atomically(target: Path, data: bytes) -> None:
        """Write to a sibling temp file, flush, then rename into place.

        A direct write to the final path leaves a partially written blob if the
        process dies mid-write, and that blob would then be served for a digest that
        no longer matches its contents. ``os.replace`` is atomic on the same
        filesystem, so a reader sees either no file or the complete file.
        """
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            with open(temporary, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)

    def _verified_read(self, candidate: Path, digest: str) -> bytes:
        try:
            data = candidate.read_bytes()
        except OSError as exc:  # pragma: no cover - defensive
            raise EvidenceRejected("EVIDENCE_UNREADABLE") from exc
        if sha256(data).hexdigest() != digest:
            raise EvidenceRejected("EVIDENCE_BLOB_MISMATCH")
        return data

    def read_bytes(self, digest: str) -> bytes:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise EvidenceRejected("EVIDENCE_DIGEST_INVALID")
        for extension in ALLOWED_MEDIA.values():
            candidate = self.root / f"{digest}{extension}"
            if candidate.exists():
                return self._verified_read(candidate, digest)
        raise EvidenceRejected("EVIDENCE_NOT_FOUND")

    def read_text(self, digest: str) -> str | None:
        """Return decoded text for text evidence, ``None`` for binary evidence."""
        try:
            data = self.read_bytes(digest)
        except EvidenceRejected:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return None
