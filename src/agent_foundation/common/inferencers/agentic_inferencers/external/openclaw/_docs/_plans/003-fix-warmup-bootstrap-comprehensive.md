# Plan 003: Comprehensive Fix for Agent Warmup/Bootstrap

**Date:** 2026-04-17
**Status:** Proposed
**Priority:** Critical — agent is non-functional on fresh sessions

---

## What Changed: The "Chief of Staff" Rewrite

Commit `5d041a3` ("Transform default workspace into Chief of Staff agent",
Apr 15 by kgrennan) rewrote all workspace-defaults files. This is the root
cause of the broken warmup.

### Three-Layer Architecture

The template files exist at three levels, each more specific:

```
Layer 1: openclaw upstream (docs/reference/templates/)
  ↓ overridden by
Layer 2: openclaw-dist-main (openshell/workspace-defaults/)
  ↓ overridden by
Layer 3: openclaw-dist working copy (openshell/sandbox/plugins/rovoclaw/workspace-defaults/)
                                     ← commit 5d041a3 rewrote this layer
```

### What Each Layer Had (Before vs After 5d041a3)

| File | Layer 1 (upstream) | Layer 2 (main branch) | Layer 3 (after 5d041a3) |
|------|-------------------|----------------------|------------------------|
| **BOOTSTRAP.md** | Conversational: "Hey, who am I? Who are you?" → figure it out together | Structured: "Your first message will be a welcome message with your name" → research TWG → greet | Complex 5-step ritual: receive name from UI or ask, 14 TWG queries, questions, write files, delete self |
| **IDENTITY.md** | Fully blank: Name/Creature/Vibe/Emoji all empty | Template with `[filled during bootstrap]` for 3 fields | Rich template: 44 lines of personality/priorities/proactivity, 3 fields still `[filled during bootstrap]` |
| **USER.md** | Simple: Name/What to call them/Pronouns/Timezone/Notes | Template with `[filled during bootstrap]` for all fields + Atlassian/Slack sections | Same template fields as main |
| **SOUL.md** | Philosophical: "Be genuinely helpful", "Have opinions", "Be resourceful" | Operational: Be friendly, be proactive, be concise, TWG instructions, identity section | Returned to upstream philosophical style + "Joining the Dots" signal synthesis + boundaries |
| **AGENTS.md** | N/A | Basic session startup + ClawGate + TWG + memory | Same structure, largely unchanged |
| **HEARTBEAT.md** | N/A | N/A (didn't exist) | **NEW** — 240 lines of proactive behavior config |

---

## Root Cause Analysis: Why the Old Flow Worked

### Main Branch Flow (WORKING through run.sh)

```
1. User installs openclaw-dist
2. run.sh runs generate_openclaw_config() → derives email from $USER
   → writes to openclaw.json as skills.entries.twg.env.TWG_USER
3. openclaw-start.sh copies workspace-defaults, starts gateway
4. User opens the UI → manually types the welcome message:
   "Your name is Claw. Research TWG and learn about me."
5. Agent reads BOOTSTRAP.md → sees it matches the expected pattern
6. Agent adopts the name, runs TWG (which uses TWG_USER internally),
   greets the user with personalized info
7. BOOTSTRAP.md says "after welcome, behave normally" — done
```

**Key facts from investigation:**
- **No code sends the welcome message automatically** — not run.sh, not
  openclaw-start.sh, not the control UI, not `openclaw onboard`
- The user was expected to **manually type** the welcome message
- `run.sh` already collects the user's email during `generate_openclaw_config()`
  and stores it as `TWG_USER` env var in openclaw.json
- The TWG skill's `--scope me` resolves to `TWG_USER` internally, so the
  agent CAN discover user identity via `scripts/twg work query --scope me`
  without knowing the email explicitly
- The system prompt contains `Authorized senders: slack:U09N0GM3V8S` (raw)
  but NOT the user's email or AAID
- **Both flows (old and new) require the user to trigger bootstrap** — the
  difference is the old flow was a simple single-message trigger while the
  new flow is a complex multi-step ritual

### New Flow (BROKEN)

```
1. User installs openclaw-dist
2. openclaw-start.sh copies new workspace-defaults (5d041a3 versions)
3. User opens UI → sends a regular message (or heartbeat fires)
4. Agent reads BOOTSTRAP.md → sees the 5-step ritual
5. Step 1 says: "Your first message may come from the setup UI with your
   name, avatar, and emoji already chosen. If so, write them..."
6. But the first message is NOT from setup UI — it's a regular user message
7. BOOTSTRAP.md says: "If the user just starts talking, ask them for name/emoji"
8. Meanwhile, USER.md has [filled during bootstrap] everywhere
9. Agent tries to ask "What should I call myself?" while user asks "prep my meeting"
10. → Agent responds: "I don't know who you are yet — USER.md is blank"
```

**Key failure:** The new BOOTSTRAP.md assumes either (a) a setup UI sends
structured data, or (b) the user will patiently go through a name-picking
ritual before doing real work. Neither happens in practice.

---

## The Five Problems (Updated from Plan 002)

### Problem 1: IDENTITY.md has empty placeholders
**Status:** FIXED in Issue 001 — defaults pre-populated (Claw 🦞)

### Problem 2: USER.md has empty placeholders with no fallback
USER.md has `[filled during bootstrap]` for every field. The agent reads it
at session startup (AGENTS.md Step 2) and finds nothing useful.

**The main branch had the same placeholders** — but it worked because
BOOTSTRAP.md's welcome flow filled them immediately via TWG in the first
message. The new BOOTSTRAP.md defers this to Step 2 (after asking the user
questions), leaving USER.md empty for the critical early interactions.

### Problem 3: BOOTSTRAP.md assumes a setup UI that doesn't exist yet
The new Step 1 says: "Your first message may come from the setup UI with
your name, avatar, and emoji already chosen." But:
- The Rovo Desktop UI doesn't have a setup flow that sends this data
- The `openclaw-control-ui` WebSocket client doesn't send setup data
- The main branch relied on a specific structured welcome message, not a
  setup UI

So the "if so, write them immediately" branch never fires, and the agent
falls through to the "ask them" branch — which breaks the user's intent
(they wanted to ask a question, not set up the agent).

### Problem 4: SOUL.md lost operational instructions
The main branch SOUL.md had explicit, operational instructions:
- "Be proactive. Don't wait to be asked."
- "Know the Atlassian ecosystem" — specific TWG usage instructions
- "On first interaction, use TWG to build an initial picture"
- "Load the twg skill from `/sandbox/.agents/skills/twg/SKILL.md`"
- Identity section with name/tone/built-by

The new SOUL.md is philosophical ("Joining the Dots", "Hold conclusions
loosely") but **lost all the operational TWG instructions**. The agent no
longer knows to proactively use TWG to discover the user on first contact.

The TWG instructions still exist in AGENTS.md (Section "Teamwork Graph"),
but the agent reads SOUL.md first (AGENTS.md Session Startup Step 1) and
SOUL.md no longer tells it what to DO — only what to BE.

### Problem 5: Memory templates have example data
Same as Plan 002 — `_example: true` entries in JSON files can confuse the
agent into treating fake data as real.

---

## Fix Plan

### Fix 0: openclaw-start.sh — Pre-populate USER.md with known identity (CRITICAL — do first)

**File:** `openshell/sandbox/openclaw-start.sh`

`openclaw-start.sh` already copies workspace-defaults (including USER.md with
placeholder fields) to `~/.openclaw/workspace/` on first boot (lines 114-135).
After that copy, it should inject the user's email — which is already
available as `$TWG_USER` in the sandbox environment (set by `run.sh`'s
`generate_openclaw_config()` and passed into the sandbox).

**Change:** After the existing workspace-defaults copy loop, add:

```bash
# ── Pre-populate USER.md with known identity ─────────────────────────
# TWG_USER is set by run.sh (e.g. tchen7@atlassian.com). If available,
# inject it into USER.md so the agent knows its human from first boot.
user_md="${WORKSPACE_DIR}/USER.md"
if [[ -n "${TWG_USER:-}" ]] && [[ -f "${user_md}" ]]; then
  # Derive name from email prefix (tchen7 → tchen7, best-effort)
  local user_prefix="${TWG_USER%%@*}"
  
  # Replace placeholder fields with known values
  sed -i.bak \
    -e "s|\[filled during bootstrap\]|${TWG_USER}|" \
    -e "s|\[from system prompt or TWG\]|${TWG_USER}|" \
    -e "s|\[discovered via TWG\]|(discover via TWG on first run)|" \
    -e "s|\[ask during bootstrap\]|(discover via TWG on first run)|" \
    -e "s|\[ask during bootstrap, or infer from activity\]|(infer from activity)|" \
    -e "s|\[discovered during bootstrap\]|(discover via TWG on first run)|" \
    -e "s|\[discover via Slack tools once connected.*\]|(check Authorized senders in system prompt)|" \
    -e "s|\[discover via Slack tools\]|(discover via Slack tools)|" \
    "${user_md}" && rm -f "${user_md}.bak"
  
  echo "==> USER.md: pre-populated with TWG_USER=${TWG_USER}"
fi

# Also inject Slack User ID if available from ownerAllowFrom in openclaw.json
slack_owner=$(python3 -c "
import json, sys
try:
    cfg = json.load(open('${HOME}/.openclaw/openclaw.json'))
    owners = cfg.get('commands',{}).get('ownerAllowFrom',[])
    for o in owners:
        if o.startswith('slack:'):
            print(o.split(':',1)[1])
            break
except: pass
" 2>/dev/null)
if [[ -n "${slack_owner}" ]] && [[ -f "${user_md}" ]]; then
  sed -i.bak \
    -e "s|(check Authorized senders in system prompt)|${slack_owner}|" \
    "${user_md}" && rm -f "${user_md}.bak"
  echo "==> USER.md: injected Slack User ID=${slack_owner}"
fi
```

**Why `openclaw-start.sh` (not `run.sh`):**
- It already handles workspace-defaults installation — this is a natural
  extension of that existing logic
- It runs every sandbox boot — if workspace is reset, USER.md gets
  re-populated on next start
- The email is already available as `$TWG_USER` in the sandbox env
- Keeps `run.sh` clean — `run.sh` handles host-side orchestration;
  writing agent workspace files is sandbox-side work

**Result:** The agent knows the user's email and Slack ID from the moment
it boots — before any message arrives, before any TWG call, before
BOOTSTRAP.md is even read. This is true zero-wait identity for email/Slack.

### Fix 0b: openclaw-start.sh — Auto-trigger bootstrap after gateway starts (CRITICAL)

**File:** `openshell/sandbox/openclaw-start.sh`

Fix 0 gives the agent email + Slack ID at boot, but NOT the user's name,
role, team, or manager — those require a TWG call. Without Fix 0b, the
agent sits idle until the user sends the first message, and only then
discovers their full identity (~10s delay).

Fix 0b sends a silent bootstrap message to the agent immediately after the
gateway starts, triggering BOOTSTRAP.md Step 0 automatically. By the time
the user opens the UI, USER.md is fully populated.

**Change:** After the gateway start and health-check in `openclaw-start.sh`,
add:

```bash
# ── Auto-trigger bootstrap ──────────────────────────────────────────
# Send a silent bootstrap message so the agent discovers its human's
# identity via TWG before the user sends their first message.
# Only fires if BOOTSTRAP.md still exists (first run / not yet bootstrapped).
bootstrap_md="${WORKSPACE_DIR}/BOOTSTRAP.md"
if [[ -f "${bootstrap_md}" ]]; then
  echo "==> Triggering auto-bootstrap (BOOTSTRAP.md exists)..."
  
  # Wait for gateway to be ready (health check)
  local retries=0
  while ! curl -sf http://localhost:18789/health >/dev/null 2>&1; do
    retries=$((retries + 1))
    if [[ $retries -ge 30 ]]; then
      echo "    Gateway not ready after 30s — skipping auto-bootstrap"
      break
    fi
    sleep 1
  done
  
  if curl -sf http://localhost:18789/health >/dev/null 2>&1; then
    curl -s -X POST http://localhost:18789/rovoclaw/chat \
      -H "Content-Type: application/json" \
      -d "{
        \"sessionKey\": \"agent:main:bootstrap\",
        \"message\": \"[SYSTEM] Auto-bootstrap: Run BOOTSTRAP.md Step 0 now. Discover your human via TWG (--scope me), populate USER.md with their name/email/role/team/manager, then continue with the rest of BOOTSTRAP.md. Do not wait for user input.\",
        \"from\": \"system\",
        \"sender\": {\"accountId\": \"system\", \"displayName\": \"system\"}
      }" >/dev/null 2>&1 &
    echo "==> Auto-bootstrap triggered (background)"
  fi
fi
```

**Key design decisions:**
- Runs in **background** (`&`) — doesn't block startup; user can open UI
  immediately while bootstrap runs in parallel
- Only fires if **BOOTSTRAP.md exists** — idempotent; won't re-trigger on
  subsequent reboots after bootstrap deletes the file
- Uses a dedicated session key (`agent:main:bootstrap`) — doesn't pollute
  the main chat session
- **Gateway health check** with 30s timeout — won't send to a dead gateway
- Message is prefixed with `[SYSTEM]` — agent can distinguish system
  triggers from user messages

**Result:** Full user identity (name, email, role, team, manager, Slack ID)
is available **~15-20 seconds after gateway starts**, before the user even
opens the UI. Combined with Fix 0 (email + Slack at boot), this gives:

| Data | When available | Source |
|------|---------------|--------|
| Email + Slack ID | **T0 (boot)** | Fix 0: bash sed |
| Name, role, team, manager | **T0+15s (auto-bootstrap)** | Fix 0b: TWG `--scope me` |

### Fix 1: BOOTSTRAP.md — Restore working welcome flow + add fallback (CRITICAL)

The main branch's welcome flow worked because it was simple and deterministic.
The new flow is creative but fragile. **Merge the best of both.**

**Key insight from investigation:** The TWG skill has `--scope me` which
resolves to the user's email via the `TWG_USER` env var (set by `run.sh`
during install). The agent doesn't need sender.accountId — it can discover
its human's identity immediately with `scripts/twg work query --scope me`.

**Changes to make:**

```markdown
# BOOTSTRAP.md — First Run Setup

_You just came online. This is your first conversation. Time to get to work._

There is no memory yet. This is a fresh workspace — memory files don't
exist until you create them. That's normal.

## Step 0 — Identify Your Human (do this FIRST, before anything else)

You don't need to ask who your human is. You can discover them immediately.

1. Load the TWG skill from `/sandbox/.agents/skills/twg/SKILL.md`
2. Run: `scripts/twg work query --scope me --since 7d --mode agent`
   This resolves "me" to your human via TWG_USER (already configured).
3. Run: `scripts/twg org-tree --scope me --up-only --mode agent`
   This gives you their name, role, team, and manager.

Write name, email, role, team, manager, and timezone to USER.md immediately.
This takes <10 seconds and unblocks everything else.

Also check your system prompt for `Authorized senders: slack:XXXXX` and
write that Slack User ID to USER.md.

## Step 1 — Figure Out Who You Are

You already have a working identity in IDENTITY.md — name, emoji, and vibe
are pre-set with sensible defaults. Use them immediately.

Your first message may come from the setup UI with a custom name, avatar,
and emoji. If so, update IDENTITY.md. Otherwise, you're Claw 🦞 — introduce
yourself and get to work. Don't ask "what should I call myself?" — just
use your default name and offer to change it later.

[...rest of existing Steps 2-5 can stay largely as-is, with the user
identity already resolved from Step 0...]
```

**Why this works:**
- Step 0 resolves user identity in ~10 seconds using TWG's `--scope me`
- **No dependency on sender.accountId** — works even on cold sessions,
  heartbeats, cron, or any trigger type
- **No dependency on a setup UI** that doesn't exist yet
- **No asking the user questions** before doing useful work
- Compatible with both the old welcome message flow AND regular first messages
- Uses infrastructure already in place (`TWG_USER` env var from `run.sh`)

### Fix 2: USER.md — Add actionable discovery instructions (HIGH)

Replace `[filled during bootstrap]` with instructions telling the agent HOW
to discover each field:

```markdown
- **Name:** (run `scripts/twg org-tree --scope me --up-only` to discover)
- **Email:** (TWG_USER is pre-configured — run `scripts/twg work query --scope me` to confirm)
- **Atlassian Account ID:** (from TWG org-tree result, or sender.accountId in inbound messages)
- **Slack User ID:** (from Authorized senders in system prompt — look for slack:XXXXX)
- **Timezone:** (from userTimezone in system prompt, or ask)
```

**Why:** Even if BOOTSTRAP.md isn't read (subagent sessions, cron sessions),
an agent reading USER.md knows exactly what to do when it finds empty fields.
Using `--scope me` means this works without any user interaction at all.

### Fix 3: SOUL.md — Restore operational TWG instructions (HIGH)

The main branch SOUL.md had critical operational content that the new version
dropped. Add back to the new SOUL.md:

```markdown
## Teamwork Graph (TWG)

You have access to the Atlassian Teamwork Graph via the `twg` skill. This is
your primary source of truth for understanding the user's work world.

Load it from: `/sandbox/.agents/skills/twg/SKILL.md`

**On first interaction with a user**, use TWG to build an initial picture:
- Look up their profile, role, team
- Find their manager and org structure
- Check recent work activity (issues, PRs, pages)
- Identify active Atlas projects and goals

**Ongoing**: Use TWG proactively to find related issues, answer questions
about teams/goals/org structure, and look up people the user mentions.
```

**Why:** SOUL.md is read first (Session Startup Step 1). Without TWG
instructions in SOUL.md, the agent doesn't know it should use TWG until
it reads AGENTS.md (Step 5), by which time it may have already responded
to the user without context.

### Fix 4: AGENTS.md — Add self-healing session startup (MEDIUM)

Update Session Startup Step 2 to handle blank USER.md:

```markdown
2. Read `USER.md` — this is who you're helping
   - **If USER.md has placeholder fields** (e.g. "[filled during bootstrap]"
     or fields are blank): Load the TWG skill and run
     `scripts/twg org-tree --scope me --up-only --mode agent` to discover
     your human's identity. Write name, email, role, team, and manager to
     USER.md before proceeding. This is a <10s operation that requires no
     user interaction.
```

**Why:** Makes every session self-healing. If USER.md is blank for ANY
reason (interrupted bootstrap, manual reset, fresh workspace), the agent
fixes it immediately. Uses `--scope me` so it works on any session type
(main, heartbeat, cron, subagent) without needing sender metadata.

### Fix 5: Memory templates — Ship clean (LOW)

Remove `_example: true` entries from all four JSON files. Keep schema
documentation, remove fake data.

**Why:** Eliminates the risk of example data being treated as real data.

---

## Implementation Order

| # | Fix | Risk if skipped | Effort | Files |
|---|-----|----------------|--------|-------|
| **0** | **openclaw-start.sh — pre-populate USER.md** | Agent has no email/Slack at boot | 15 min | `openclaw-start.sh` |
| **0b** | **openclaw-start.sh — auto-trigger bootstrap** | Agent doesn't know name/role until first user message | 10 min | `openclaw-start.sh` |
| 1 | BOOTSTRAP.md — Step 0 `--scope me` + fallback | Agent wastes time asking questions | 20 min | `workspace-defaults/BOOTSTRAP.md` |
| 2 | USER.md — actionable instructions | Sessions stay broken if bootstrap fails | 10 min | `workspace-defaults/USER.md` |
| 3 | SOUL.md — restore TWG instructions | Agent doesn't use TWG proactively | 10 min | `workspace-defaults/SOUL.md` |
| 4 | AGENTS.md — self-healing session startup | No recovery from partial bootstrap | 5 min | `workspace-defaults/AGENTS.md` |
| 5 | Memory templates — clean examples | Agent may use fake data | 10 min | `memory/*.json` |

**Total effort:** ~80 minutes. Fixes 0+0b together give full identity at
boot. Fixes 0-4 should be one commit. Fix 5 can be same or separate.

---

## What NOT to Change

- **HEARTBEAT.md** — this is net-new and useful. Keep it.
- **IDENTITY.md** — already fixed in Issue 001. Keep the rich personality content.
- **Skills (daily-brief, meeting-prep)** — these are good additions. Keep them.
- **Memory system (4 JSON files)** — the architecture is sound. Just clean the templates.
- **AGENTS.md structure** — the session startup, memory docs, skills table, heartbeat config are all good. Only add the self-healing fallback.

The goal is surgical: **restore what broke (user identity discovery) while
keeping what improved (personality, skills, heartbeats, memory system).**

---

## Expected Outcome

### Before (broken)
```
T0:  openclaw-start.sh copies workspace-defaults (USER.md with placeholders)
T1:  Gateway starts
T2:  User sends "prep my Monday meeting"
T3:  Agent reads USER.md → "[filled during bootstrap]" everywhere
T4:  Agent: "I don't know who you are — USER.md is blank"
```

### After (with Fix 0 + 0b — full identity at boot, zero user action)
```
T0:   openclaw-start.sh copies workspace-defaults
T0+:  openclaw-start.sh injects TWG_USER=tchen7@atlassian.com + Slack
      ID=U09N0GM3V8S into USER.md via sed — pure bash, no agent
T1:   Gateway starts, IDENTITY.md has defaults (Claw 🦞)
T1+:  Fix 0b: auto-bootstrap fires → sends system message to agent
T2:   Agent reads BOOTSTRAP.md Step 0 → runs `--scope me` via TWG
T3:   Agent writes name=Tony Chen, role, team, manager to USER.md (~15s)
T4:   Agent deletes BOOTSTRAP.md → bootstrap complete
...
T10:  User opens UI, sends "prep my Monday meeting"
T11:  Agent reads USER.md → fully populated ✅ → proceeds immediately
```

### After (with Fix 4 — self-healing on reboots)
```
T0:  Sandbox reboots (crash, update, etc.)
T0+: openclaw-start.sh skips USER.md copy (already exists with real data)
T0+: Fix 0b skips (BOOTSTRAP.md already deleted) — no re-trigger
T1:  Gateway starts
T2:  User message → agent reads USER.md → already populated ✅
     (If somehow blank: AGENTS.md self-healing runs `--scope me` to fix)
```

**Identity availability timeline:**

| Data | When available | Source |
|------|---------------|--------|
| Email | **T0 (boot)** | Fix 0: `openclaw-start.sh` injects `TWG_USER` |
| Slack User ID | **T0 (boot)** | Fix 0: extracted from `openclaw.json` |
| Name, role, team, manager | **T0+15s (auto-bootstrap)** | Fix 0b: auto-trigger → TWG `--scope me` |
| Full identity (all fields) | **T0+15s — before user even opens UI** | Fix 0 + Fix 0b combined |

---

## Related

- **Issue 001** — IDENTITY.md placeholder fix (DONE)
- **Plan 002** — Initial warmup fix plan (superseded by this plan)
- **Commit 5d041a3** — "Transform default workspace into Chief of Staff agent"
- **Main branch comparison** — `openclaw-dist-main/openshell/workspace-defaults/`
