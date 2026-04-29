# Plan 005: Fix Timezone Discovery During Warmup

**Date:** 2026-04-17
**Status:** Proposed
**Priority:** Medium — agent doesn't know user's timezone, breaks heartbeat,
daily-brief, meeting-prep, and any time-aware feature.

---

## Problem Statement

After warmup, the agent doesn't know the user's timezone. Live workspace
shows:

```
USER.md:
- Timezone: [ask — active at 23:43 UTC on a Friday, unclear]
```

The agent is reduced to guessing from activity patterns (e.g. "you posted
at 23:43 UTC, that could be PDT or anything"). For a Pacific Time user,
the agent reports `Friday April 17, 2026 — 7:20 PM (UTC)` instead of
`Friday April 17, 2026 — 12:20 PM PDT` — making "Monday meeting", "tomorrow",
"morning brief" all calculated against the wrong wall-clock.

### Symptoms

- Daily brief mode (morning/midday/afternoon) selected against UTC, not user's TZ
- "Monday meeting" calendar lookups wrong by 17+ hours
- Heartbeat working-hours check uses UTC → fires at 3am user time
- Meeting-prep "30 minutes before" trigger fires at the wrong wall clock
- Any "today", "tomorrow", "yesterday" reasoning is off by a day

---

## Root Cause

**Three independent gaps stack up:**

### Gap 1: Container TZ is hard-coded to UTC

```bash
$ docker exec ... sh -c 'echo $TZ; date; cat /etc/timezone'
TZ env:                  # empty
Sat Apr 18 00:19:23 UTC 2026
Etc/UTC
```

The sandbox base image sets `Etc/UTC` and never gets overridden. `run.sh`
(host) and `openclaw-env.sh` (container init) have no `TZ` handling.

### Gap 2: Host-side `run.sh` doesn't pass user's TZ

The host knows the user's local timezone (from macOS `systemsetup -gettimezone`
or just `date +%Z`), but `run.sh` never reads it or passes it as a `TZ`
environment variable when creating the sandbox.

### Gap 3: USER.md template instructions are wrong

```markdown
- **Timezone:** [from userTimezone in system prompt, or ask]
```

There is **no `userTimezone` field in the system prompt** (verified — not
in `openclaw.json` or any other source). The agent looks for it, doesn't
find it, then asks the user — but only when prompted, not during warmup.
And the warmup doesn't ask either, so it just stays as the placeholder
`[ask — active at 23:43 UTC on a Friday, unclear]`.

### Stack-up effect

- Container TZ = UTC → `date` returns UTC
- No `TZ` passed in → can't fall back to host TZ
- USER.md template references nonexistent `userTimezone` → agent has nothing
  to fill it from
- BOOTSTRAP.md Step 0 mentions timezone as a thing to discover but provides
  no command to discover it

So even if the agent runs Step 0 correctly, it has no source of truth for
timezone other than asking the user — which it doesn't do during silent
auto-bootstrap.

---

## Fix Plan

### Fix 1: `run.sh` — Capture host TZ and pass to sandbox (HIGH)

**File:** `openshell/run.sh`

In the sandbox creation flow, capture the host's IANA timezone and pass it
as a Docker env var:

```bash
# Detect host timezone (macOS + Linux)
USER_TZ="${TZ:-}"
if [[ -z "$USER_TZ" ]]; then
  if [[ "$(uname)" == "Darwin" ]]; then
    USER_TZ="$(systemsetup -gettimezone 2>/dev/null | sed 's/Time Zone: //' || readlink /etc/localtime | sed 's|.*/zoneinfo/||')"
  else
    USER_TZ="$(cat /etc/timezone 2>/dev/null || readlink /etc/localtime | sed 's|.*/zoneinfo/||')"
  fi
  USER_TZ="${USER_TZ:-America/Los_Angeles}"  # fallback
fi
export USER_TZ
```

Then ensure `USER_TZ` propagates into the openshell sandbox (likely via the
existing env var injection mechanism — same path as `TWG_USER`).

### Fix 2: `openclaw-start.sh` — Apply TZ inside container (HIGH)

**File:** `openshell/sandbox/openclaw-start.sh`

After workspace defaults install (and before the gateway starts), set the
container TZ:

```bash
# ── Set container timezone from host ─────────────────────────────────
if [[ -n "${USER_TZ:-}" ]]; then
  if [[ -f "/usr/share/zoneinfo/${USER_TZ}" ]]; then
    export TZ="${USER_TZ}"
    # Also try to update /etc/localtime if we have permission
    if [[ -w /etc/localtime ]] || command -v sudo >/dev/null 2>&1; then
      ln -sf "/usr/share/zoneinfo/${USER_TZ}" /etc/localtime 2>/dev/null || true
      echo "${USER_TZ}" > /etc/timezone 2>/dev/null || true
    fi
    echo "  Timezone set to: ${USER_TZ} ($(date +%Z))"
  else
    echo "  WARN: USER_TZ='${USER_TZ}' not found in zoneinfo, keeping UTC"
  fi
fi
```

`export TZ` ensures all subsequent `date` calls and gateway processes inherit
the right TZ. The `/etc/localtime` symlink is a best-effort upgrade — works
in dev containers, may be no-op in restricted PVC.

### Fix 3: `openclaw-start.sh` — Inject TZ into USER.md (HIGH)

**File:** `openshell/sandbox/openclaw-start.sh`

In the existing USER.md pre-population block (currently injects TWG_USER
email and Slack ID), also inject the timezone:

```bash
# Inject timezone if known
if [[ -n "${USER_TZ:-}" ]] && grep -q '\[from userTimezone\|\[ask\b' "${user_md}" 2>/dev/null; then
  sed -i "s|- \*\*Timezone:\*\* \[.*\]|- **Timezone:** ${USER_TZ}|" "${user_md}"
  echo "  USER.md: injected Timezone=${USER_TZ}"
fi
```

### Fix 4: `USER.md` template — Update timezone field instructions (MEDIUM)

**File:** `workspace-defaults/USER.md`

Replace the misleading instruction:
```markdown
- **Timezone:** [from userTimezone in system prompt, or ask]
```

With:
```markdown
- **Timezone:** [auto-populated by openclaw-start from host system; falls back to America/Los_Angeles]
```

This documents the actual source of truth so the agent doesn't go looking
for a nonexistent `userTimezone` field.

### Fix 5: `BOOTSTRAP.md` — Add timezone fallback to Step 0 (LOW)

**File:** `workspace-defaults/BOOTSTRAP.md`

In Step 0, after running TWG `--scope me`, add a fallback for missing TZ:

```markdown
**If timezone is still empty after Step 0:**
Run `date +%Z` in the shell to get the current TZ abbreviation. If it
returns "UTC", the host TZ wasn't injected — leave Timezone field as
"unknown — please tell me your timezone" and ask the user on first
interaction.
```

### Fix 6: Add fingerprint marker to is_old_template() (LOW)

**File:** `openshell/sandbox/openclaw-start.sh`

So the new USER.md template gets installed on PVC-persisted old workspaces:

```bash
USER.md)
  ...existing checks...
  # Old version with broken userTimezone reference
  grep -q "from userTimezone in system prompt" "$file" 2>/dev/null && return 0
  ;;
```

---

## Implementation Order

| # | Fix | Effort | Files | Required for warmup TZ? |
|---|-----|--------|-------|--------------------------|
| 1 | `run.sh` — detect + pass host TZ | 10 min | `openshell/run.sh` | ✅ Required |
| 2 | `openclaw-start.sh` — apply TZ in container | 5 min | `openshell/sandbox/openclaw-start.sh` | ✅ Required |
| 3 | `openclaw-start.sh` — inject TZ into USER.md | 5 min | (same) | ✅ Required |
| 4 | `USER.md` template — fix instructions | 2 min | `workspace-defaults/USER.md` | Recommended |
| 5 | `BOOTSTRAP.md` — TZ fallback in Step 0 | 5 min | `workspace-defaults/BOOTSTRAP.md` | Optional |
| 6 | Fingerprint marker for USER.md | 2 min | `openshell/sandbox/openclaw-start.sh` | Recommended (PVC) |

**Total: ~30 minutes, 4 files, one commit.**

**Recommended minimum:** Fixes 1, 2, 3, 4. (5 and 6 are belt-and-suspenders.)

---

## Expected Outcome

### Before
```
USER.md:
- Timezone: [ask — active at 23:43 UTC on a Friday, unclear]

Agent: "Hi! It's Friday April 17, 2026 — 7:20 PM (UTC)"
       (User actual time: 12:20 PM PDT)

Daily brief mode: morning (because UTC < 11am)
       (User actual time: afternoon)
```

### After
```
USER.md:
- Timezone: America/Los_Angeles

Agent: "Hi! It's Friday April 17, 2026 — 12:20 PM PDT"
       (Matches user wall clock)

Daily brief mode: midday (correct)
Heartbeat working-hours check: respects PDT
Meeting prep "Monday meeting": resolves to user's Monday, not UTC's
```

---

## Validation

After changes, verify in the live container:

```bash
# 1. Container TZ is set
docker exec ... sh -c 'echo "$TZ"; date'   # should show TZ + local wall clock

# 2. USER.md has timezone
grep "Timezone" /sandbox/.openclaw/workspace/USER.md
# Expected: - **Timezone:** America/Los_Angeles

# 3. New session uses correct time
# In OpenClaw UI: "what time is it?"
# Expected: matches user's local wall clock, NOT UTC
```

---

## What We Are NOT Doing

- ❌ Asking the user for their timezone during warmup. Goes against the
  "zero-wait identity" goal — host TZ is the right source.
- ❌ Inferring TZ from activity patterns. Unreliable; breaks for travelers
  but more importantly the host already knows the right answer.
- ❌ Storing TZ in openclaw.json. Container `TZ` env var + USER.md is
  enough; openclaw.json shouldn't carry user-specific runtime config.

---

## Related

- **Issue 001** — IDENTITY.md placeholder fix (DONE)
- **Plan 003** — Comprehensive warmup/bootstrap fix (DONE)
- **Plan 004** — Skill auto-invocation via trigger phrases (DONE)
- **This plan completes the "zero-wait warmup" trio:** identity (Plan 003)
  + skill discovery (Plan 004) + timezone (Plan 005). After all three, the
  agent boots with full identity, can route requests to skills correctly,
  and reasons about time in the user's local frame.
