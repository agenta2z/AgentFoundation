"""What guards a BTA call's checkpoint root (plan v8 §5.11, §5.12).

A call holds an exclusive lease on ``<checkpoint_root>/.bta_execution.lock`` from the
moment its call record is taken until its invocation's ledger closes, so no two
holders — threads, processes or overlapping calls — ever read or write one checkpoint
tree at once.

Under the lease, ``bta_manifest.json`` (written atomically) holds the call's identity
header — input, definition and invocation arguments — and, once the worker list is
built, the committed effective plan the workers are rebuilt from on resume.
"""

import json
import os
import tempfile
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from agent_foundation.common.inferencers.run_context.errors import (
    InvocationContractError,
)
from agent_foundation.common.inferencers.run_context.resume_identity import (
    identity_bytes,
    identity_digest,
    qualified_name,
    ResumeIdentityUnavailableError,
)
from rich_python_utils.io_utils.file_lock import FileLock

LEASE_FILE = ".bta_execution.lock"


class BtaWorkspaceBusyError(InvocationContractError):
    """Another holder has this BTA's checkpoint root (§5.11). An invocation
    contract error, so never retried."""


def _refuse_unsafe_root(root: str) -> None:
    """A checkpoint root is never the filesystem root, ``$HOME`` or a repo root."""
    real = os.path.realpath(root)
    home = os.path.realpath(os.path.expanduser("~"))
    if real in (os.path.sep, home) or any(
        os.path.isdir(os.path.join(real, marker)) for marker in (".git", ".hg", ".sl")
    ):
        raise ValueError(
            f"refusing to lease checkpoint root {root!r}: it is the filesystem root, "
            f"the home directory or a repository root"
        )


class CheckpointLease:
    """The exclusive hold one BTA call has on its checkpoint root.

    ``acquire`` never waits: a held root raises ``BtaWorkspaceBusyError`` before the
    call reads or writes anything there. ``close`` releases it (the invocation's
    ledger calls it, last); process death releases it too.
    """

    def __init__(self, root: str) -> None:
        self.root = root
        self._lock: Optional[FileLock] = None

    @property
    def held(self) -> bool:
        return self._lock is not None and self._lock.held

    def acquire(self) -> "CheckpointLease":
        _refuse_unsafe_root(self.root)
        os.makedirs(self.root, exist_ok=True)
        lock = FileLock(os.path.join(self.root, LEASE_FILE))
        if not lock.acquire(timeout=0):
            raise BtaWorkspaceBusyError(
                f"BTA checkpoint root {self.root!r} is in use by another call, "
                f"thread or process; give each concurrent call its own workspace"
            )
        self._lock = lock
        return self

    def close(self) -> None:
        lock, self._lock = self._lock, None
        if lock is not None:
            lock.release()


# --- The resume identity manifest (§5.12) ------------------------------------------

MANIFEST_FILE = "bta_manifest.json"
SCHEMA_VERSION = 1
CHECKPOINT_VERSION = 1

TRUST_LEGACY = "trust_legacy"
RESUME_IDENTITY_POLICIES = ("verify", TRUST_LEGACY)


class BtaResumeIdentityMismatch(InvocationContractError):
    """The checkpoints at this root were written for another input, definition or
    invocation arguments."""


class BtaResumeCorruptionError(InvocationContractError):
    """The manifest or the checkpoint tree is not one this protocol writes."""


class UnverifiedLegacyBtaResumeError(InvocationContractError):
    """Checkpoints without a manifest: nothing proves they belong to this call."""


def _digest_or_reason(value: Any, path: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        return identity_digest(value, path), None
    except ResumeIdentityUnavailableError as exc:
        return None, str(exc)


def build_header(
    inference_input: Any, definition: Any, arguments: Any
) -> Dict[str, Any]:
    """The identity header: the input's type, length and digest, the definition's
    digest and the invocation arguments' digest. A part without an identity is
    ``None``, and ``unverifiable`` says why."""
    input_digest, input_reason = _digest_or_reason(inference_input, "input")
    definition_digest, definition_reason = _digest_or_reason(definition, "definition")
    arguments_digest, arguments_reason = _digest_or_reason(arguments, "arguments")
    length = None
    if input_digest is not None:
        length = len(identity_bytes(inference_input, "input"))
    return {
        "input": {
            "type": qualified_name(type(inference_input)),
            "length": length,
            "digest": input_digest,
        },
        "definition": definition_digest,
        "arguments": arguments_digest,
        "unverifiable": input_reason or definition_reason or arguments_reason,
    }


def header_mismatches(stored: Mapping[str, Any], current: Mapping[str, Any]) -> list:
    """The header parts that differ (``input``, ``definition``, ``arguments``)."""
    return [
        part
        for part in ("input", "definition", "arguments")
        if stored.get(part) != current.get(part)
    ]


def read_manifest(root: str) -> Optional[Dict[str, Any]]:
    """The manifest at ``root``, or ``None`` when there is none."""
    path = os.path.join(root, MANIFEST_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as exc:
        raise BtaResumeCorruptionError(
            f"unreadable BTA manifest {path!r}: {exc}"
        ) from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("checkpoint_version") != CHECKPOINT_VERSION
        or not isinstance(manifest.get("header"), dict)
        or not isinstance(manifest.get("plan"), (dict, type(None)))
    ):
        raise BtaResumeCorruptionError(f"BTA manifest {path!r} has an unknown shape")
    return manifest


def write_manifest(
    root: str, header: Mapping[str, Any], plan: Optional[Mapping[str, Any]]
) -> None:
    """Write the manifest atomically: a temp file, ``fsync``, ``os.replace``."""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "header": dict(header),
        "plan": None if plan is None else dict(plan),
    }
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, MANIFEST_FILE)
    fd, tmp = tempfile.mkstemp(dir=root, prefix=f".{MANIFEST_FILE}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def remove_manifest(root: str) -> None:
    try:
        os.remove(os.path.join(root, MANIFEST_FILE))
    except FileNotFoundError:
        pass


def _has_files(path: str, skip: Callable[[str], bool] = lambda entry: False) -> bool:
    if not os.path.isdir(path):
        return False
    for entry in os.listdir(path):
        if skip(entry):
            continue
        full = os.path.join(path, entry)
        if not os.path.isdir(full) or _has_files(full):
            return True
    return False


def _is_protocol_file(entry: str) -> bool:
    # A temp file a manifest write left behind when its process died.
    return entry in (LEASE_FILE, MANIFEST_FILE) or entry.startswith(
        f".{MANIFEST_FILE}."
    )


def has_root_artifacts(root: str) -> bool:
    """Whether ``root`` holds anything besides the lease and the manifest."""
    return _has_files(root, skip=_is_protocol_file)


def has_worker_results(worker_checkpoint_dirs: Iterable[str]) -> bool:
    return any(_has_files(path) for path in worker_checkpoint_dirs)


def plan_record(sub_queries: Any, worker_nodes: Iterable[str]) -> Dict[str, Any]:
    """The committed effective plan: the sub-queries the workers are built from, in
    order, and the worker nodes they produced. Sub-queries that don't survive a
    JSON round trip unchanged can't be replayed: the plan then records why instead
    of them."""
    workers = list(worker_nodes)
    try:
        replayable = json.loads(json.dumps(sub_queries))
    except (TypeError, ValueError) as exc:
        reason = f"sub-queries are not JSON values: {exc}"
    else:
        if replayable == sub_queries:
            return {"sub_queries": replayable, "workers": workers}
        reason = "sub-queries change in a JSON round trip (tuples, non-string keys)"
    return {"unavailable": reason, "workers": workers}
