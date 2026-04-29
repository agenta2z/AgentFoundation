# Plan 004: Fix Skill Discovery & Auto-Invocation

**Date:** 2026-04-17
**Status:** Proposed
**Priority:** Medium-High — skills exist but require explicit user invocation

---

## Problem Statement

The `meeting-prep` and `daily-brief` skills exist and work correctly, but the
agent only invokes them when the user explicitly says "use the meeting-prep
skill" or "use daily-brief skill." Natural language requests like
"prep my Monday meeting" or "what should I prepare?" do NOT trigger the
skills — the agent does the work using TWG + Slack directly without loading
the skill framework.

### Observed Behavior

```
User: "prep my Monday meeting"
Agent: [does ad-hoc TWG queries] → produces decent but flat answer

User: "prep my Monday meeting. Use the meeting-prep skill."
Agent: [loads skill, follows framework] → produces structured, prioritized brief
```

The skill version is meaningfully better (clear #1 priority, owner table,
sharper talking points), but users have to know skills exist AND know to
ask for them by name.

---

## Root Cause

**The rovoclaw skill descriptions don't follow the OpenClaw convention used
by `twg` and `agentic-search` — they describe WHAT the skill does but not
WHEN to use it.**

### Convention used by other OpenClaw skills

The `twg` skill description (which the agent invokes correctly):

```yaml
description: >
  Use `scripts/twg` to query and manage Atlassian TeamWork Graph Jira,
  Confluence, Atlas, Loom, Bitbucket, Google docs, Sharepoint and other
  work data.
  Trigger when a user asks for cross-product activity, entity details
  (issue/page/goal/project/pr/video), team or user insights, across
  Atlassian products or create/update actions through terminal commands.
```

Two-part structure:
1. Sentence 1 — WHAT it does
2. **Sentence 2 — `Trigger when a user asks for [concrete patterns]`**

The `agentic-search` skill follows the same convention with even stronger
trigger language: *"Use this skill whenever someone asks you to find, look
up, search for, or research anything in company knowledge bases — even if
they don't explicitly say 'search'."*

### How the rovoclaw skills break the convention

**`meeting-prep/SKILL.md`:**
```yaml
description: >
  Prepare context and talking points before meetings. Gathers attendee
  context from TWG, memory, and live sources, then synthesizes into an
  actionable brief with relationship awareness and a draft opener.
  Under 200 words.
```
*Tells WHAT it does and HOW it works. No "Trigger when..." clause.*

**`daily-brief/SKILL.md`:**
```yaml
description: >
  Generate morning, midday, and afternoon briefings that synthesize signals
  across Jira, Confluence, Loom, Slack, and memory into an actionable
  summary. Runs 3× daily on schedule, or on demand.
```
*Tells WHAT and WHEN it runs (schedule), but no user-facing trigger phrases.*

Without a "Trigger when..." clause, the agent's intent matcher has no
pattern to match user requests against. So requests like "prep my meeting"
fall through to ad-hoc TWG/Slack queries instead of routing to the skill.

### AGENTS.md duplication

**Important:** `AGENTS.md` is hand-written and contains its own copy of the
skill descriptions in a skills table. Changing `SKILL.md` descriptions does
NOT auto-update `AGENTS.md` — both must be updated.

Current AGENTS.md skills table:
```
| `daily-brief`  | .../daily-brief/SKILL.md  | Generate morning/midday/afternoon briefings |
| `meeting-prep` | .../meeting-prep/SKILL.md | Prepare context and talking points before meetings |
```

These descriptions are even more truncated than the SKILL.md versions and
also lack any trigger info.

---

## Fix Plan

### Fix 1: Rewrite `meeting-prep/SKILL.md` description (HIGH)

**File:** `openshell/sandbox/plugins/rovoclaw/skills/meeting-prep/SKILL.md`

Add a "Trigger when..." clause following the `twg` convention:

```yaml
description: >
  Prepare context and talking points before a meeting. Gathers attendee
  context from TWG, memory, and live sources, then synthesizes into an
  actionable, prioritized brief with talking points and a draft opener.
  Trigger when a user asks to prep for, brief on, or get ready for a
  meeting (e.g., "prep my Monday meeting", "brief me on the war room",
  "what do I need to know before standup", "help me prep for [meeting]").
```

### Fix 2: Rewrite `daily-brief/SKILL.md` description (HIGH)

**File:** `openshell/sandbox/plugins/rovoclaw/skills/daily-brief/SKILL.md`

```yaml
description: >
  Generate morning, midday, and afternoon briefings that synthesize signals
  across Jira, Confluence, Loom, Slack, and memory into an actionable
  summary. Runs 3× daily on schedule.
  Trigger when a user asks for a daily summary, morning brief, status
  digest, or "what's happening today" / "what do I need to know" /
  "catch me up" / "morning brief".
```

### Fix 3: Update `AGENTS.md` skills table to match (HIGH)

**File:** `openshell/sandbox/plugins/rovoclaw/workspace-defaults/AGENTS.md`

Replace the truncated descriptions with the enriched trigger-bearing versions:

```markdown
| Skill | Location | What it does + when to use |
|-------|----------|----------------------------|
| `daily-brief`  | `/sandbox/openclaw-plugins/rovoclaw/skills/daily-brief/SKILL.md`  | Morning/midday/afternoon briefings. Trigger when user asks for a daily summary, morning brief, status digest, "catch me up", or "what's happening today". |
| `meeting-prep` | `/sandbox/openclaw-plugins/rovoclaw/skills/meeting-prep/SKILL.md` | Prep brief for an upcoming meeting. Trigger when user asks to prep for, brief on, or get ready for any meeting (e.g., "prep my Monday meeting", "brief me on the war room"). |
```

---

## Implementation Order

| # | Fix | Effort | Files |
|---|-----|--------|-------|
| 1 | Rewrite `meeting-prep/SKILL.md` description | 5 min | 1 file |
| 2 | Rewrite `daily-brief/SKILL.md` description | 5 min | 1 file |
| 3 | Update `AGENTS.md` skills table | 5 min | 1 file |

**Total: ~15 minutes, 3 files, one commit.**

**Token cost:** ~150 tokens added across the 3 files. Negligible per session.

---

## What We Are NOT Changing

- ❌ `autoLoad: true` — would cost ~7K tokens per session for ~3.5% context
  window. Not needed; the description-level fix achieves the same outcome.
- ❌ SOUL.md "Skills First" preamble — redundant with AGENTS.md changes;
  same agent reads both files at session start.
- ❌ Any logic changes to openclaw runtime — this is a pure content fix.

---

## Expected Outcome

### Before

```
User: "prep my Monday meeting"
Agent: [no trigger match → falls back to ad-hoc TWG/Slack queries]
       → Decent but flat answer
```

### After

```
User: "prep my Monday meeting"
Agent: [matches "prep" + "meeting" against meeting-prep trigger phrase]
       → Loads meeting-prep skill
       → Follows framework: identify → classify → gather → synthesize
       → Returns prioritized brief with #1 Priority, blockers table,
         owner attribution, sharp talking points
```

---

## Validation

After changes, test these queries WITHOUT mentioning the skill name:

| Query | Expected behavior |
|-------|-------------------|
| "prep my Monday RovoClaw meeting" | Loads meeting-prep ✅ |
| "what's happening today?" | Loads daily-brief ✅ |
| "brief me on my next meeting" | Loads meeting-prep ✅ |
| "morning brief" | Loads daily-brief ✅ |
| "catch me up" | Loads daily-brief ✅ |
| "what should I know before standup?" | Loads meeting-prep ✅ |

If any still don't trigger correctly, add the failing pattern to the
"Trigger when..." clause in the corresponding SKILL.md and AGENTS.md.

---

## Related

- **Issue 001** — IDENTITY.md placeholder fix (DONE)
- **Plan 003** — Comprehensive warmup/bootstrap fix (DONE)
- **Observation:** Even without the skill, OpenClaw produces ~95% of the
  same content (TWG + Slack capability gathers the right info). The skill
  adds shape/prioritization, not new data. This is a "polish" fix, not a
  "correctness" fix — but the polish meaningfully improves output for
  meeting prep and daily brief specifically.
