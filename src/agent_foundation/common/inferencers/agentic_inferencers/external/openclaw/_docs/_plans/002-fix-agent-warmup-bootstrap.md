# Plan 002: Fix Agent Warmup / Bootstrap Flow

**Date:** 2026-04-17
**Status:** Proposed
**Priority:** High — agent is non-functional on fresh sessions until bootstrap completes

---

## Problem Statement

When OpenClaw starts with a fresh workspace, the agent doesn't know who the
user is and can't perform any useful work until a full bootstrap completes.
The bootstrap flow is entirely agent-driven (relies on the LLM reading
BOOTSTRAP.md and executing a 5-step ritual), which creates multiple failure
modes:

### Observed Symptoms

1. **"I don't know who you are"** — USER.md contains only `[filled during bootstrap]` placeholders; agent can't look up meetings, prepare briefs, or do anything user-specific
2. **No memory files** — `memory/people-index.json`, `tasks.json`, etc. contain example entries only (`_example: true`)
3. **Nameless agent** — IDENTITY.md has `[to be chosen during bootstrap]` for name/vibe/emoji (partially fixed — see Issue 001)
4. **ClawGate down** — separate issue but compounds the bootstrap failure
5. **No AAID/email in system prompt** — the system prompt contains authorized sender IDs (`slack:U09N0GM3V8S`) but no user email or AAID, so the agent can't self-serve user identity without TWG

### Root Cause Analysis

The warmup architecture has **four distinct problems**:

#### Problem 1: IDENTITY.md ships with empty placeholders
**Status: Fixed** (Issue 001 — defaults now pre-populated with Claw 🦞)

#### Problem 2: USER.md ships with empty placeholders
USER.md is the agent's knowledge of the human it serves. Every field is a
`[filled during bootstrap]` placeholder. Unlike IDENTITY.md (which is about
the agent's personality), USER.md contains **factual data** that can and
should be pre-populated from known sources:

- **Atlassian Account ID** — available from the Slack `ownerAllowFrom` config (`slack:U09N0GM3V8S` → can be resolved via TWG)
- **Email** — discoverable from AAID via TWG
- **Slack User ID** — already in `commands.ownerAllowFrom` as `slack:U09N0GM3V8S`
- **Timezone** — already in the system prompt via `agents.defaults.userTimezone`

But even without pre-population, the agent should be able to bootstrap USER.md
from the first inbound message, which carries `sender.accountId` and
`sender.displayName`. The problem is that BOOTSTRAP.md defers this to Step 2
(after a slow TWG discovery phase), leaving USER.md empty for 30-90 seconds.

#### Problem 3: Memory templates contain example data, not empty state
The four memory JSON files ship with `_example: true` entries. The agent is
instructed in BOOTSTRAP.md Step 4 to "remove the example entry." But:

- If bootstrap doesn't complete, examples persist forever
- The agent may try to use example data as real data
- Memory files should ship clean (empty arrays) with the schema/docs as
  `_schema`/`_description` fields only

#### Problem 4: Bootstrap is a single monolithic agent flow with no checkpoints
The entire bootstrap is one continuous agent run:
1. Get name/avatar → 2. Run 14 TWG queries → 3. Ask questions → 4. Write files → 5. Delete BOOTSTRAP.md

If any step fails, hangs, or the session is interrupted:
- No partial progress is saved
- BOOTSTRAP.md isn't deleted → next session re-runs from scratch
- Files remain in template state
- There's no way for subsequent sessions to know bootstrap failed vs. hasn't started

---

## Fix Plan

### Fix A: Pre-populate USER.md with discoverable defaults (HIGH — do first)

**File:** `workspace-defaults/USER.md`

Replace `[filled during bootstrap]` placeholders with instructions the agent
can act on immediately, plus any data extractable from config:

```markdown
# USER.md — About Your Human

_Auto-populated on first run. Updated as you learn more._

- **Name:** (discover via TWG on first message — check sender.accountId)
- **What to call them:** (ask them, or use first name from TWG)
- **Email:** (discover via TWG using sender.accountId from the first inbound message)
- **Atlassian Account ID:** (from sender.accountId in inbound message payload)
- **Role:** (discover via TWG)
- **Team:** (discover via TWG)
- **Manager:** (discover via TWG)
- **Timezone:** (from system prompt userTimezone, or ask)
- **Working hours:** (infer from activity, or ask)
- **Calendar ID:** (usually same as email)

## Atlassian Sites

- **Jira:** (discover via TWG or system prompt)
- **Confluence:** (discover via TWG or system prompt)

## Slack (if connected)

- **Slack User ID:** (check commands.ownerAllowFrom in system prompt for slack:XXXXX)
- **Slack Display Name:** (discover via Slack tools)
```

**Key insight:** The parenthetical instructions tell the agent *how* to fill
each field rather than just saying "filled during bootstrap." This makes
USER.md self-documenting even before bootstrap runs.

### Fix B: Update BOOTSTRAP.md Step 1 to write USER.md immediately (HIGH)

**File:** `workspace-defaults/BOOTSTRAP.md`

Add a new "Step 0" or modify Step 1 to extract user identity from the first
inbound message **before** doing anything else:

```markdown
## Step 0 — Identify Your Human (do this FIRST, before anything else)

Your first inbound message carries sender info. Extract it immediately:
- `sender.accountId` → write to USER.md as Atlassian Account ID
- `sender.displayName` → write to USER.md as Name (tentative)
- Check your system prompt for `Authorized senders: slack:XXXXX` → write as Slack User ID

Then run ONE quick TWG lookup to resolve the rest:
  twg user get --account-id <accountId> --mode agent

Write name, email, role, team, and manager to USER.md immediately.
This takes <5 seconds and gives you a working USER.md for everything else.
```

**Why this matters:** Currently USER.md stays blank until Step 4 (after the
slow TWG discovery in Step 2). By extracting identity in Step 0, the agent
has a working USER.md within 5 seconds of the first message, even if the
full bootstrap takes 60+ seconds.

### Fix C: Clean up memory template files (MEDIUM)

**Files:** `workspace-defaults/memory/*.json`

Remove example entries. Ship with empty arrays and schema metadata only:

```json
{
  "_schema": "people-index",
  "_description": "Key people in the user's work world...",
  "people": []
}
```

Same for `project-index.json` (`"projects": []`), `tasks.json` (`"tasks": []`),
`updates.json` (`"updates": []`). Keep the `_schema`, `_description`,
`_statuses`, `_priorities`, `_sources`, and `_types` documentation fields —
remove only the `_example: true` entries.

**Why:** Eliminates the risk of example data being treated as real data, and
removes the BOOTSTRAP.md Step 4 instruction to "remove the example entry"
which is a fragile LLM-dependent cleanup step.

### Fix D: Add bootstrap progress tracking (LOW — future improvement)

**File:** `workspace-defaults/AGENTS.md` + `workspace.ts` integration

Add a `## Bootstrap Status` section to AGENTS.md that the agent updates as
it completes each step:

```markdown
## Bootstrap Status

- [x] Identity set (IDENTITY.md)
- [ ] User identified (USER.md)
- [ ] TWG discovery complete
- [ ] Memory files initialized
- [ ] First daily brief run
- [ ] BOOTSTRAP.md deleted
```

This gives subsequent sessions visibility into partial bootstrap state, and
prevents re-running completed steps. It also allows the UI to show bootstrap
progress.

**Note:** This is a future improvement. Fixes A-C eliminate the most critical
failures without requiring progress tracking.

### Fix E: Make non-bootstrap sessions resilient to empty USER.md (MEDIUM)

**File:** `workspace-defaults/AGENTS.md`

Update the "Session Startup" section to handle the case where USER.md is
still a template:

```markdown
## Session Startup

Before doing anything else:

1. Read `SOUL.md` — this is who you are
2. Read `USER.md` — this is who you're helping
   - **If USER.md still has placeholder fields:** Check the sender info from
     the current message. Run `twg user get --account-id <sender.accountId>`
     to resolve their identity. Write results to USER.md immediately.
3. Read `IDENTITY.md` — your name, personality, and style
4. Read `memory/YYYY-MM-DD.md` (today + yesterday) for recent context
5. **If in MAIN SESSION** (direct chat with your human): Also read `MEMORY.md`
```

This makes every session self-healing — if USER.md is blank for any reason
(interrupted bootstrap, fresh workspace, manual reset), the agent fixes it
on the spot.

---

## Implementation Order

| Priority | Fix | Files Changed | Effort |
|----------|-----|---------------|--------|
| 1 | **Fix A** — USER.md with actionable instructions | `workspace-defaults/USER.md` | 10 min |
| 2 | **Fix B** — BOOTSTRAP.md Step 0 for immediate identity | `workspace-defaults/BOOTSTRAP.md` | 15 min |
| 3 | **Fix C** — Clean memory templates | `workspace-defaults/memory/*.json` | 10 min |
| 4 | **Fix E** — Self-healing session startup | `workspace-defaults/AGENTS.md` | 10 min |
| 5 | **Fix D** — Bootstrap progress tracking | Future PR | 30 min |

Fixes A-C should be done together as one commit. Fix E can be the same
commit or a follow-up. Fix D is a future improvement.

---

## Expected Outcome After Fixes

### Before (current state)
```
T0:  Gateway starts
T1:  User sends "prep my Monday meeting"
T2:  Agent reads USER.md → "[filled during bootstrap]" → "I don't know who you are"
T3:  Agent reads memory/people-index.json → finds example data
T4:  Agent gives up: "I need to know who you are first"
```

### After (with fixes)
```
T0:  Gateway starts, IDENTITY.md has working defaults (Claw 🦞)
T1:  User sends "prep my Monday meeting"
T2:  Agent reads USER.md → sees actionable instructions → extracts sender.accountId
T3:  Agent runs `twg user get --account-id <id>` → writes name/email/role to USER.md (5s)
T4:  Agent reads memory/*.json → clean empty arrays, no confusion
T5:  Agent proceeds with meeting prep using the now-populated USER.md
```

**Race window reduced from 30-90s → 5s**, and the agent is functional
(with partial user context) from the first message.

---

## Related Issues

- **Issue 001** — IDENTITY.md placeholder fields (FIXED)
- **Gateway protocol mismatch** — `_stream_gateway()` in openclaw_inferencer.py uses wrong frame format (separate issue)
- **ClawGate down** — `localhost:13377` not responding (separate infrastructure issue)
