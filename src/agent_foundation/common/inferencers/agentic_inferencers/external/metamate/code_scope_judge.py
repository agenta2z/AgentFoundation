# pyre-strict

"""Metamate code-search scope judge — the first ``@agentic_function`` application.

Metamate's remote ``code_search`` runs under a ~512 MiB HHVM per-request cap; the
unscoped default (``repo:"all"``) fans out to ACL-restricted corpora and dies with
a fatal "Unauthorized" give-up. This judge decides a search corpus (never ``all``;
default ``fbsource``) plus fixed read-discipline, conveyed as a prompt directive.

Security model: the model authors ONLY a closed semantic judgment
(``{scope, confidence, rationale}``); trusted Python maps that to the operational
dual-tool vocabulary via a host-owned table. A prompt-injected "search all / load
everything" cannot widen scope — the model's output alphabet has no ``all``, no
free-text corpus, no path, and no discipline string. :func:`parse_code_scope_judgment`
rejects any smuggled operational key and fails closed to ``fbsource``.

Two tool families exist (verified): a string-``repo`` family (Unified / Enhanced /
Local / Composite) and an enum-``corpus`` family (BigGrep / Apex). ``fbcode`` is a
legal ``corpus`` but NOT a legal ``repo``, so the directive must name both
spellings — a single scope token would be wrong for one family.
"""

from __future__ import annotations

import enum
import logging
import math
from typing import Any, Awaitable, Callable

import attr
from agent_foundation.common.inferencers.agentic_functions import (
    agentic_function,
    AgenticOutput,
    ParseError,
)

_logger: logging.Logger = logging.getLogger(__name__)


class CodeSemanticScope(str, enum.Enum):
    """The ONLY thing the model chooses — a closed semantic area, never operational."""

    FBCODE = "fbcode"
    FBSOURCE = "fbsource"
    WWW = "www"
    CONFIGERATOR = "configerator"
    INSTAGRAM = "instagram"
    NON_CODE = "non_code"
    UNKNOWN = "unknown"


class Repo(str, enum.Enum):
    """The string-``repo`` tool family's value set (Unified / Enhanced / Local / Composite).

    Deliberately has NO ``ALL`` member: ``all`` is unrepresentable by construction.
    """

    FBSOURCE = "fbsource"
    WWW = "www"
    CONFIGERATOR = "configerator"
    INSTAGRAM = "instagram"


class Corpus(str, enum.Enum):
    """The enum-``corpus`` tool family's value set (BigGrep / Apex)."""

    FBSOURCE = "fbsource"
    FBCODE = "fbcode"
    WWW = "www"
    FBOBJC = "fbobjc"
    FBSOURCE_REST = "fbsource_rest"
    CONFIGERATOR = "configerator"
    CONFIGERATOR_MATERIALIZED = "configerator-materialized"
    NUCLEUS_14 = "aosp-nucleus-14-sandcastle"


READ_DISCIPLINE: str = (
    "Locate with search first; read only tight line ranges; never load whole files."
)


# Closed, host-authored mapping. repo / corpora / paths are ALL host-derived — the
# model never reaches past a CodeSemanticScope into this table.
_POLICY: dict[CodeSemanticScope, tuple[Repo, tuple[Corpus, ...], tuple[str, ...]]] = {
    CodeSemanticScope.FBCODE: (Repo.FBSOURCE, (Corpus.FBCODE,), ("fbcode",)),
    CodeSemanticScope.FBSOURCE: (Repo.FBSOURCE, (Corpus.FBSOURCE,), ()),
    CodeSemanticScope.WWW: (Repo.WWW, (Corpus.WWW,), ("www",)),
    CodeSemanticScope.CONFIGERATOR: (
        Repo.CONFIGERATOR,
        (Corpus.CONFIGERATOR,),
        ("configerator",),
    ),
    CodeSemanticScope.INSTAGRAM: (
        Repo.INSTAGRAM,
        (Corpus.FBSOURCE,),
        ("fbcode/instagram",),
    ),
    CodeSemanticScope.NON_CODE: (Repo.FBSOURCE, (Corpus.FBSOURCE,), ()),
    CodeSemanticScope.UNKNOWN: (Repo.FBSOURCE, (Corpus.FBSOURCE,), ()),
}

# Keys the model must NEVER author. If any appears in the reply, it tried to author
# operational scope directly — reject and fail closed.
_OPERATIONAL_KEYS: frozenset[str] = frozenset(
    {
        "repo",
        "repos",
        "corpus",
        "corpora",
        "path",
        "paths",
        "regex",
        "command",
        "read_discipline",
        "discipline",
    }
)

_ALLOWED_KEYS: frozenset[str] = frozenset({"scope", "confidence", "rationale"})


@attr.s(auto_attribs=True, frozen=True)
class CodeScopeJudgment:
    """The model's CLOSED output after validation — no operational fields."""

    scope: CodeSemanticScope
    confidence: float
    rationale: str


@attr.s(auto_attribs=True, frozen=True)
class CodeSearchScope:
    """Operational scope — host-DERIVED from a :class:`CodeScopeJudgment` via ``_POLICY``.

    ``repo`` / ``corpora`` / ``paths`` / ``read_discipline`` are all host-owned;
    only ``confidence`` / ``rationale`` carry the model's (validated) judgment.
    """

    repo: Repo = Repo.FBSOURCE
    corpora: tuple[Corpus, ...] = (Corpus.FBSOURCE,)
    paths: tuple[str, ...] = ()
    read_discipline: str = READ_DISCIPLINE
    confidence: float = 0.0
    rationale: str = ""

    @classmethod
    def default(cls) -> "CodeSearchScope":
        """The safe fallback: ``fbsource`` / ``(FBSOURCE,)``, never ``all``."""
        return cls()

    def to_directive(self) -> str:
        """Host-owned search directive naming BOTH tool spellings + read-discipline."""
        corpora = ", ".join(c.value for c in self.corpora)
        lines = [
            "## Code-search scope (host-enforced — follow exactly):",
            f"- For string-`repo` tools (Unified/Enhanced/Local/Composite), use "
            f"repo: {self.repo.value}",
            f"- For enum-`corpus` tools (BigGrep/Apex), use corpus: {corpora}",
        ]
        if self.paths:
            lines.append(f"- Prefer paths under: {', '.join(self.paths)}")
        lines.append(f"- Read discipline: {self.read_discipline}")
        lines.append(
            '- Never use repo:"all" or an unscoped corpus: it fans out to '
            "access-controlled corpora and fails with Unauthorized."
        )
        return "\n".join(lines)


def parse_code_scope_judgment(response: object) -> CodeScopeJudgment:
    """STRICT validation of the model's ``scope_decision`` dict (stage-2a).

    Fails closed (raises :class:`ParseError`, which the decorator retries then
    falls back on) if the model authored any operational key, a ``bool``
    confidence, an out-of-range / non-finite confidence, an unexpected key, or an
    invalid enum value.
    """
    if not isinstance(response, dict):
        raise ParseError("scope_decision is not a JSON object")
    keys = set(response.keys())
    leaked = _OPERATIONAL_KEYS & keys
    if leaked:
        raise ParseError(
            f"model emitted operational field(s) it must not author: {sorted(leaked)}"
        )
    unexpected = keys - _ALLOWED_KEYS
    if unexpected:
        raise ParseError(f"unexpected key(s) in scope_decision: {sorted(unexpected)}")
    if "scope" not in response:
        raise ParseError("scope_decision missing required key 'scope'")
    try:
        scope = CodeSemanticScope(str(response["scope"]))
    except ValueError as e:
        raise ParseError(f"invalid scope value: {response['scope']!r}") from e
    confidence = response.get("confidence", 0.0)
    # bool is a subclass of int: reject an injected `"confidence": true` outright
    # rather than let float(True) == 1.0 slip through as full confidence.
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ParseError("confidence must be a finite number (not a bool)")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
        raise ParseError(f"confidence out of [0, 1]: {confidence}")
    rationale = str(response.get("rationale", ""))
    return CodeScopeJudgment(scope=scope, confidence=confidence, rationale=rationale)


def scope_from_code_judgment(judgment: CodeScopeJudgment) -> CodeSearchScope:
    """Trusted mapping from a validated judgment to operational scope (stage-2b)."""
    repo, corpora, paths = _POLICY[judgment.scope]
    return CodeSearchScope(
        repo=repo,
        corpora=corpora,
        paths=paths,
        confidence=judgment.confidence,
        rationale=judgment.rationale,
    )


def code_search_scope_parser(response: object) -> CodeSearchScope:
    """``scope_from_code_judgment ∘ parse_code_scope_judgment`` as one reusable callable.

    Registered in the shared target registry as ``CodeSearchScopeParser`` so any
    ``@agentic_function`` can reference it by that alias — on its own for a
    ``scope_decision`` dict, or after :func:`_extract_code_scope_block` in a parser
    sequence to go straight from an :class:`AgenticOutput` to a ``CodeSearchScope``.
    """
    return scope_from_code_judgment(parse_code_scope_judgment(response))


def _extract_code_scope_block(output: AgenticOutput) -> dict:
    """Stage-1 parser: pull the labeled ```json scope_decision fence as a dict.

    Raises :class:`ParseError` (via :meth:`AgenticOutput.json`) if the block is
    absent or not valid JSON — which the decorator retries, then falls back.
    """
    return output.json("scope_decision")


# Plugboard routes to Claude with CAT auth, so the judge needs no API key and no
# per-model MetaGen entitlement (every mg-api key tried was denied on every Claude).
# ``usecase-dev-ai`` is a published pipeline this identity holds ``pipeline_execute``
# on, and it serves the current Sonnet.
JUDGE_MODEL_ID: str = "claude-sonnet-5"
JUDGE_PIPELINE: str = "usecase-dev-ai"

# ``None`` = do not send ``temperature`` at all. The current Sonnet deployment
# rejects it ("`temperature` is deprecated for this model"), and the judge has no
# need for it: the task is a closed-set classification and the decode fails closed
# anyway. It is set on the inferencer rather than via ``infer_kwargs`` because
# ``PlugboardApiInferencer._infer``/``_ainfer`` already pass
# ``temperature=self.temperature`` into ``generate_text``, so also forwarding it
# through ``**_inference_args`` would raise "got multiple values for keyword
# argument 'temperature'".
JUDGE_TEMPERATURE: float | None = None


def _build_judge_inferencer() -> Any:
    """Build the judge's Plugboard inferencer (a zero-arg factory, built on first call).

    The framework default is ``ClaudeApiInferencer``, which talks to the public
    Anthropic API and needs ``ANTHROPIC_API_KEY`` — unusable from inside Meta.
    Plugboard is the Meta-internal path, authenticated by CAT.

    The import is deliberately deferred into this factory: Plugboard pulls in
    generated Thrift and its native ``folly`` extension, and this module is
    imported by ``metamate/__init__``, so importing it eagerly would make the
    whole metamate package unimportable wherever those are unavailable.
    """
    from agent_foundation.common.inferencers.api_inferencers.plugboard.plugboard_api_inferencer import (
        PlugboardApiInferencer,
    )

    return PlugboardApiInferencer(
        model_id=JUDGE_MODEL_ID,
        pipeline=JUDGE_PIPELINE,
        temperature=JUDGE_TEMPERATURE,
        max_retry=2,
        total_timeout_seconds=60,
    )


@agentic_function(
    inferencer=_build_judge_inferencer,
    template="prompt_templates/code_scope_judge.jinja2",
    run_context_slot="code_scope_judge",
    parser=_extract_code_scope_block,
    parse_max_retries=1,
    fallback=CodeSearchScope.default,
)
async def judge_code_scope(task: str, *, response: dict) -> CodeSearchScope:
    """Decide the code-search scope for an (untrusted) Metamate task.

    Stage 1 (decorator ``parser=``) extracts the ```json scope_decision fence to a
    dict; stage 2 (this body) strictly validates it and maps it through the
    host-owned policy. A malformed or injected reply fails closed to
    :meth:`CodeSearchScope.default` (``fbsource``) — never ``all``.
    """
    return scope_from_code_judgment(parse_code_scope_judgment(response))


async def resolve_scope_directive(
    judge: Callable[[str], Awaitable[Any]], task: str
) -> str:
    """Return the host-owned search directive for ``task``, failing closed.

    Any judge failure degrades to the default (``fbsource``) directive rather
    than propagating: the judge is an optimization for the call it scopes, and
    must never be able to break it.

    The decorator cannot close this gap itself — its ``fallback=`` is reached
    only for ``retry_on`` (i.e. ``ParseError``), so a transport or configuration
    failure (an unreachable backend, a missing MetaGen key) escapes the agentic
    function entirely. A judge returning something without ``to_directive()`` is
    caught here too, since the seam accepts any callable.
    """
    try:
        scope = await judge(task)
        return scope.to_directive()
    except Exception:
        _logger.warning(
            "code scope judge failed; falling back to the default scope",
            exc_info=True,
        )
        return CodeSearchScope.default().to_directive()
