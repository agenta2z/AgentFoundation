.. _error-handling:

================
Error Recovery
================

This document covers OpenClaw's error recovery mechanisms, including
model fallback, auth profile rotation, thinking level adaptation,
session auto-reset, and iteration guards.  The primary source files
are:

- ``src/agents/model-fallback.ts`` -- model fallback chain execution
- ``src/agents/pi-embedded-runner/run.ts`` -- embedded runner with
  retry and compaction logic
- ``src/agents/failover-error.ts`` -- failover error classification
- ``src/agents/pi-embedded-helpers.ts`` -- error classification
  helpers

.. contents:: On this page
   :local:
   :depth: 2


Model Fallback Chain
=====================

When the primary model fails with an eligible error, OpenClaw
automatically tries alternative models from a configured fallback
chain.

Candidate Resolution
---------------------

``resolveFallbackCandidates()``
(``src/agents/model-fallback.ts``:173-234) builds the ordered
candidate list:

.. code-block:: text

   1. Primary model (user or agent-configured)
   2. Configured fallbacks (agents.defaults.model.fallbacks)
   3. Global default model (agents.defaults.model.primary)

Each candidate is de-duplicated by ``provider/model`` key and
optionally filtered against a configured allowlist:

.. code-block:: typescript
   :caption: src/agents/model-fallback.ts:64-87

   function createModelCandidateCollector(
     allowlist: Set<string> | null | undefined
   ): {
     candidates: ModelCandidate[];
     addCandidate: (
       candidate: ModelCandidate,
       enforceAllowlist: boolean,
     ) => void;
   }

.. note::

   When the user is running a non-default model override (e.g., via
   ``!model``), the configured fallback chain is skipped.  On failure,
   the system falls back directly to the configured global default
   (line 210-213).


Run with Fallback
------------------

``runWithModelFallback()``
(``src/agents/model-fallback.ts``:280-410) executes the retry loop:

.. code-block:: typescript

   for (let i = 0; i < candidates.length; i += 1) {
     const candidate = candidates[i];
     // 1. Check auth profile cooldown
     // 2. Attempt the run
     // 3. On success: return result
     // 4. On failover-eligible error: record attempt, continue
     // 5. On abort or context overflow: rethrow immediately
   }
   // All candidates exhausted: throw summary error

**Error classification** determines whether to retry:

- **Failover-eligible** (retry with next candidate):
  Rate-limit (429), server error (500, 502, 503), timeout,
  authentication error, billing error, overloaded.
- **Non-retriable** (rethrow immediately):
  ``AbortError`` (user cancellation), context overflow
  (should be handled by compaction, not model switching).

.. code-block:: typescript
   :caption: src/agents/model-fallback.ts:367-369

   if (isLikelyContextOverflowError(errMessage)) {
     throw err;  // Don't try a smaller-context model
   }


Auth Profile Rotation
======================

When a model call fails with a rate-limit or auth error, the
**auth profile** that was used is placed into cooldown.  Subsequent
retries use a different profile:

.. code-block:: typescript
   :caption: src/agents/model-fallback.ts:306-344

   const profileIds = resolveAuthProfileOrder({
     cfg, store: authStore, provider: candidate.provider,
   });
   const isAnyProfileAvailable = profileIds.some(
     (id) => !isProfileInCooldown(authStore, id)
   );
   if (profileIds.length > 0 && !isAnyProfileAvailable) {
     // All profiles in cooldown for this provider
     // Try probing if primary and near cooldown expiry
   }

The cooldown system is managed by
``src/agents/auth-profiles/usage.ts``:

- ``markAuthProfileCooldown()`` -- set a cooldown timestamp.
- ``isProfileInCooldown()`` -- check if profile is in cooldown.
- ``calculateAuthProfileCooldownMs()`` -- exponential backoff.
- ``clearAuthProfileCooldown()`` -- manual cooldown reset.
- ``clearExpiredCooldowns()`` -- periodic cleanup.

**Probing**: For the primary model, even when all profiles are in
cooldown, the system periodically probes to detect recovery:

.. code-block:: typescript
   :caption: src/agents/model-fallback.ts:246-270

   const PROBE_MARGIN_MS = 2 * 60 * 1000;  // 2 minutes
   const MIN_PROBE_INTERVAL_MS = 30_000;    // 30 seconds

   function shouldProbePrimaryDuringCooldown(params: {
     isPrimary: boolean;
     hasFallbackCandidates: boolean;
     now: number;
     // ...
   }): boolean {
     // Probe when cooldown expiry is near or already past
     return params.now >= soonest - PROBE_MARGIN_MS;
   }


Thinking Level Fallback
=========================

When a model fails with a rate-limit or capacity error and is
using extended thinking (``high`` or ``xhigh``), the system
automatically reduces the thinking level:

.. code-block:: typescript
   :caption: src/agents/pi-embedded-helpers.ts (pickFallbackThinkingLevel)

   // xhigh -> high -> medium -> low -> off
   export function pickFallbackThinkingLevel(
     currentLevel: ThinkLevel
   ): ThinkLevel | undefined

This is applied in the retry loop of
``runEmbeddedPiAgent()`` -- if the error is a rate-limit and thinking
is enabled, the system retries with a lower thinking level before
falling back to a different model.


Session Auto-Reset
===================

When the conversation context exceeds the model's context window
and auto-compaction fails, the session is automatically reset:

.. code-block:: typescript
   :caption: src/agents/pi-embedded-runner/run.ts (approximate)

   if (isCompactionFailureError(error)) {
     // Context still too large after compaction attempt.
     // Reset the session to start fresh.
     await resetSession(sessionFile);
     // Retry with empty context.
   }

The auto-reset flow:

1. Compaction is attempted first (summarize old messages).
2. If compaction reduces context below the limit, the run resumes.
3. If compaction fails (context still too large), the session
   transcript is archived and cleared.
4. A ``[System Message]`` is injected to inform the agent that the
   session was auto-reset.
5. The run restarts with an empty context.


Iteration Guard
================

To prevent infinite tool loops, the embedded runner enforces a
maximum iteration count.  Each tool-call-and-result cycle counts as
one iteration.

The limit is typically derived from the agent configuration or a
sensible default.  When the limit is reached:

1. The current run is terminated.
2. A warning message is appended to the conversation.
3. The model's last partial output is returned as the response.


Tool Error Handling
====================

When a tool call throws an error, the error is caught and
converted into a tool result rather than crashing the run:

.. code-block:: typescript

   // Simplified flow in pi-agent-core
   try {
     result = await tool.execute(params);
   } catch (err) {
     result = {
       isError: true,
       content: [{ type: "text", text: err.message }],
     };
   }

The error result is appended to the conversation, allowing the
model to see the error and decide how to recover (retry, skip, or
inform the user).

**Special error types**:

- ``ToolInputError`` (status 400) -- invalid parameters.
  The model typically corrects its input and retries.
- ``ToolAuthorizationError`` (status 403) -- sender not authorized.
  The model informs the user that the action is restricted.


Oversized Tool Results
-----------------------

When a tool returns an excessively large result (e.g., reading a
very large file), the runner truncates it to prevent context overflow:

.. code-block:: typescript
   :caption: src/agents/pi-embedded-runner/tool-result-truncation.ts

   truncateOversizedToolResultsInSession(session, maxTokens);

The truncation:

- Detects tool results that exceed a configurable threshold.
- Replaces them with a truncation marker:
  ``[truncated: output exceeded context limit]``.
- Preserves the first and last portions of the output when possible.

The sub-agent system prompt instructs sub-agents to handle these
markers gracefully:

.. code-block:: text

   If you see `[compacted: tool output removed to free context]`
   or `[truncated: output exceeded context limit]`, assume prior
   output was reduced. Re-read only what you need using smaller
   chunks.


Image Model Fallback
======================

Image analysis has its own fallback chain via
``runWithImageModelFallback()``
(``src/agents/model-fallback.ts``:412-469):

.. code-block:: typescript

   const candidates = resolveImageFallbackCandidates({
     cfg, defaultProvider: DEFAULT_PROVIDER, modelOverride,
   });

The image fallback chain is configured separately from the main
model fallback via ``agents.defaults.imageModel``:

.. code-block:: yaml
   :caption: Example config

   agents:
     defaults:
       imageModel:
         primary: "anthropic/claude-sonnet-4-5"
         fallbacks:
           - "openai/gpt-4o"
           - "google/gemini-2.5-pro"


Error Summary on Exhaustion
============================

When all fallback candidates are exhausted, a summary error is
thrown with details about each attempt:

.. code-block:: text

   All models failed (3):
     anthropic/claude-opus-4-6: 429 Too Many Requests (rate_limit)
     | anthropic/claude-sonnet-4-5: 429 Too Many Requests (rate_limit)
     | openai/gpt-4o: 500 Internal Server Error (server_error)

This summary is surfaced to the user as an error message, giving
visibility into what was tried and why it failed.
