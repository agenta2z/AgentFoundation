"""Golden-file harness for the invocation-scoped runtime refactor.

Goldens pin today's observable behaviour (workspace trees, session-log type
sequences, getter values, call records) so that later refactor commits can
prove byte-compatibility or show exactly which tagged behaviour changed.

Regenerate a golden only with local pytest::

    AF_UPDATE_GOLDENS=1 ~/af_pytest.sh test/agent_foundation/run_context/<test>

Use one :class:`Normalizer` per golden: numbered placeholders (``<UUID1>``,
``<ID2>``) are assigned in first-appearance order, so equal volatile values stay
equal and distinct ones stay distinct (e.g. session-id continuity across two
calls is visible in the golden).
"""

import difflib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

GOLDEN_DIR = Path(__file__).parent / "goldens"
UPDATE_ENV = "AF_UPDATE_GOLDENS"

DEFAULT_VOLATILE_KEYS = frozenset(
    {
        "time",
        "timestamp",
        "created_at",
        "updated_at",
        "start_time",
        "end_time",
        "elapsed",
        "elapsed_seconds",
        "duration",
        "duration_seconds",
        "size_bytes",
    }
)

MULTI_KEY = "<multi>"

_TS = "<TS>"
_VOLATILE = "<VOLATILE>"
_NAME_RE = re.compile(r"[a-z0-9_]+(/[a-z0-9_]+)*")

# (pattern, placeholder stem, numbered). Order matters: UUIDs before bare hex.
# SHA-256 digests (the BTA resume manifest) cover qualified names, which differ
# between buck and pytest module paths.
_PATTERNS: Tuple[Tuple["re.Pattern[str]", str, bool], ...] = (
    (re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])"), "SHA", True),
    (
        re.compile(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
        ),
        "UUID",
        True,
    ),
    (re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{32}(?![0-9a-fA-F])"), "HEX", True),
    (
        re.compile(r"(?<![0-9A-Za-z])\d{8}_\d{6}(?:_[0-9a-f]{8})?(?![0-9A-Za-z])"),
        "TS",
        False,
    ),
    (
        re.compile(
            r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
            r"(?:Z|[+-]\d{2}:?\d{2})?"
        ),
        "TS",
        False,
    ),
    (re.compile(r"0x[0-9a-fA-F]{6,}"), "ADDR", True),
    (re.compile(r"(?<=[A-Za-z0-9_]-)[0-9a-f]{8}(?![0-9a-zA-Z])"), "ID", True),
)

# (regex, replacement) rules applied after ``_PATTERNS``: the random suffix of a
# log part file (``.parts/<Type>/<TS>_<stem>_<hex8>.<ext>``) is not a per-run
# identity.
_SUBSTITUTIONS: Tuple[Tuple[str, str], ...] = (
    (
        r"(\.parts/(?:[^/\s\"']+/)*[^/\s\"']*_)[0-9a-f]{8}(\.[0-9A-Za-z]+)(?![\w.])",
        r"\1<PART>\2",
    ),
)


class Normalizer:
    """Replace volatile substrings (paths, ids, timestamps) with placeholders.

    ``roots`` maps a placeholder (``"<WS>"``) to an absolute path; both the path
    and its realpath are replaced, longest first. ``extra`` adds
    ``(regex, placeholder)`` rules applied after the built-in ones.
    """

    def __init__(
        self,
        roots: Optional[Mapping[str, Any]] = None,
        *,
        volatile_keys: frozenset = DEFAULT_VOLATILE_KEYS,
        extra: Sequence[Tuple[str, str]] = (),
    ) -> None:
        pairs = []
        for placeholder, root in (roots or {}).items():
            for variant in {str(root), os.path.realpath(str(root))}:
                pairs.append((variant.rstrip(os.sep) or os.sep, placeholder))
        self._roots = sorted(pairs, key=lambda p: len(p[0]), reverse=True)
        self._volatile_keys = volatile_keys
        self._extra = [(re.compile(p), r) for p, r in (*_SUBSTITUTIONS, *extra)]
        self._numbers: Dict[Tuple[str, str], int] = {}

    def text(self, s: str, *, numbered: bool = True) -> str:
        for root, placeholder in self._roots:
            s = s.replace(root, placeholder)
        for pattern, stem, is_numbered in _PATTERNS:
            s = pattern.sub(
                lambda m, st=stem, n=is_numbered: self._placeholder(
                    st, m.group(0), n and numbered
                ),
                s,
            )
        for pattern, replacement in self._extra:
            s = pattern.sub(replacement, s)
        return s

    def value(self, obj: Any) -> Any:
        """Normalize a JSON-compatible value; raise ``TypeError`` otherwise."""
        if isinstance(obj, str):
            return self.text(obj)
        if obj is None or isinstance(obj, (bool, int, float)):
            return obj
        if isinstance(obj, Path):
            return self.text(str(obj))
        if isinstance(obj, Mapping):
            return self._mapping(obj)
        if isinstance(obj, (list, tuple)):
            return [self.value(v) for v in obj]
        if isinstance(obj, (set, frozenset)):
            return sorted((self.value(v) for v in obj), key=_sort_key)
        raise TypeError(
            f"golden value of type {type(obj).__name__} is not JSON-compatible; "
            "project it to plain data before normalizing"
        )

    def _placeholder(self, stem: str, raw: str, numbered: bool) -> str:
        if stem == "TS":
            return _TS
        if not numbered:
            return f"<{stem}>"
        key = (stem, raw)
        if key not in self._numbers:
            self._numbers[key] = 1 + sum(1 for s, _ in self._numbers if s == stem)
        return f"<{stem}{self._numbers[key]}>"

    def _mapping(self, obj: Mapping) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, val in obj.items():
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                raise TypeError(f"golden mapping key {key!r} is not JSON-compatible")
            norm_key = self.text(str(key))
            norm_val = self._keyed_value(str(key), val)
            _merge(out, norm_key, norm_val)
        return out

    def _keyed_value(self, key: str, val: Any) -> Any:
        if key in self._volatile_keys and isinstance(val, (str, int, float)):
            return _VOLATILE
        return self.value(val)


def _sort_key(v: Any) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def _merge(out: Dict[str, Any], key: str, val: Any) -> None:
    """Insert ``val``; entries whose keys collide after normalization become a
    sorted multiset under ``{MULTI_KEY: [...]}`` so no data is dropped."""
    if key not in out:
        out[key] = val
        return
    existing = out[key]
    if isinstance(existing, dict) and set(existing) == {MULTI_KEY}:
        entries = existing[MULTI_KEY] + [val]
    else:
        entries = [existing, val]
    out[key] = {MULTI_KEY: sorted(entries, key=_sort_key)}


def _is_log_path(rel: str) -> bool:
    return rel.startswith("logs/") or "/logs/" in rel


def _read_pickle(path: str, norm: Normalizer) -> Any:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    try:
        return {"pickle": type(obj).__name__, "value": norm.value(obj)}
    except TypeError:
        return {"pickle": type(obj).__name__, "repr": norm.text(repr(obj))}


def _read_json_lines(text: str, norm: Normalizer) -> List[Any]:
    return [norm.value(json.loads(line)) for line in text.splitlines() if line]


def file_entry(path: str, norm: Normalizer) -> Any:
    """Normalized content of one workspace file."""
    if path.endswith(".pkl"):
        return _read_pickle(path, norm)
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        return "<binary>"
    try:
        if path.endswith(".jsonl"):
            return {"jsonl": _read_json_lines(text, norm)}
        if path.endswith(".json"):
            return {"json": norm.value(json.loads(text))}
    except json.JSONDecodeError:
        pass
    return norm.text(text)


def _walk(root: str) -> List[Tuple[str, str]]:
    """``(relpath, kind)`` for every file and symlink; symlinked dirs are not
    descended."""
    found: List[Tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in list(dirnames):
            if os.path.islink(os.path.join(dirpath, name)):
                dirnames.remove(name)
                found.append((os.path.join(dirpath, name), "link"))
        for name in filenames:
            full = os.path.join(dirpath, name)
            found.append((full, "link" if os.path.islink(full) else "file"))
    return [(os.path.relpath(p, root).replace(os.sep, "/"), k) for p, k in found]


def _ordered(root: str, rels: List[str], norm: Normalizer) -> List[str]:
    """Sort by unnumbered normalized path, then by modification time.

    Files that differ only in a volatile id (two loggers' session logs in one
    directory after a resume) are ordered by when they were written, never by
    the random id, so numbered placeholders are assigned deterministically.
    """

    def key(rel: str) -> Tuple[str, int, str]:
        mtime = os.lstat(os.path.join(root, rel)).st_mtime_ns
        return (norm.text(rel, numbered=False), mtime, rel)

    return sorted(rels, key=key)


def workspace_tree(
    root: Any,
    norm: Normalizer,
    *,
    content_for: Optional[Callable[[str], bool]] = None,
) -> Dict[str, Any]:
    """Normalized ``{relpath: content}`` for every file under ``root``.

    Symlinks record their normalized target. ``content_for(relpath)`` selects
    which files record content; by default everything except session logs,
    whose contract is the type sequence (:func:`session_log_types`).

    Ids in paths are numbered from the ordered paths before any content is read,
    so a file that lists paths in a random order (the aggregation manifest)
    cannot change the numbering.
    """
    root = str(root)
    wants = content_for or (lambda rel: not _is_log_path(rel))
    kinds = dict(_walk(root))
    ordered = _ordered(root, list(kinds), norm)
    for rel in ordered:
        norm.text(rel)
    out: Dict[str, Any] = {}
    for rel in ordered:
        kind = kinds[rel]
        full = os.path.join(root, rel)
        if kind == "link":
            val: Any = {"symlink": norm.text(os.readlink(full))}
        else:
            val = file_entry(full, norm) if wants(rel) else "<present>"
        _merge(out, norm.text(rel), val)
    return out


def session_log_types(root: Any, norm: Normalizer) -> Dict[str, List[str]]:
    """``{normalized log relpath: [record type, ...]}`` for every session log
    under ``root``. Each log file belongs to one logger id, so the per-file
    order is that source's own order."""
    root = str(root)
    logs = [
        rel
        for rel, kind in _walk(root)
        if kind == "file" and rel.endswith(".jsonl") and "/session/" in f"/{rel}"
    ]
    out: Dict[str, List[str]] = {}
    for rel in _ordered(root, logs, norm):
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            types = [json.loads(line).get("type") for line in f if line.strip()]
        _merge(out, norm.text(rel), types)
    return out


def render(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def check_golden(name: str, data: Any) -> None:
    """Compare ``data`` (already normalized) with ``goldens/<name>.json``;
    rewrite the file instead when ``AF_UPDATE_GOLDENS`` is set."""
    assert _NAME_RE.fullmatch(name), f"bad golden name {name!r}"
    path = GOLDEN_DIR / f"{name}.json"
    rendered = render(data)
    if os.environ.get(UPDATE_ENV):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        return
    assert path.exists(), (
        f"missing golden {path}; regenerate locally with "
        f"{UPDATE_ENV}=1 ~/af_pytest.sh <this test>"
    )
    expected = path.read_text(encoding="utf-8")
    if json.loads(rendered) == json.loads(expected):
        return
    diff = "".join(
        difflib.unified_diff(
            render(json.loads(expected)).splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile=f"golden/{name}",
            tofile="actual",
        )
    )
    raise AssertionError(f"golden {name} differs:\n{diff}")


def live_branch_label(key: Any) -> str:
    """A leaf connection store's ``(handle scope, path)`` branch key as a golden
    mapping key; host scope ids are hex and normalize to numbered placeholders."""
    scope, path = key
    return f"{scope} {path}"
