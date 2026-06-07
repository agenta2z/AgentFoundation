#!/usr/bin/env bash
# scripts/check_dev_docs_present.sh
#
# Guardrail: on the dev_xinli_2601 branch, the _docs/ tree (especially the
# workflows_and_sop/ plan series) MUST be present. If it is missing on disk,
# it almost certainly means the branch was hard-reset to origin/main (which
# does not track _docs/), silently dropping ~50k lines of design plans.
#
# History: this exact regression happened on 2026-06-03 (reflog: '4d4c76d
# ... reset: moving to origin/main') and was fixed by commit 9bddfd4. This
# script exists so the same mistake is caught instantly next time, instead
# of being noticed days later.
#
# Exit codes:
#   0 = OK (or not on dev branch — script is a no-op elsewhere)
#   1 = DROP DETECTED — _docs/ missing on dev_xinli_2601
#
# Used by:
#   - .git/hooks/post-checkout (installed via scripts/install-git-hooks.sh)
#   - Optional CI step on dev_xinli_2601

set -euo pipefail

# Find the repo root (so the script works regardless of CWD)
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
  # Not in a git repo — silent no-op
  exit 0
fi
cd "${REPO_ROOT}"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"

# Only guard the dev branch. On main (or detached HEAD), do nothing.
if [[ "${CURRENT_BRANCH}" != "dev_xinli_2601" ]]; then
  exit 0
fi

# Canary file — the one we know was lost in the 2026-06-03 incident.
CANARY="_docs/_plan/workflows_and_sop/sop_runtime_enablement_plan.md"

# Two checks:
#   (a) the canary file exists in the current commit's tree (catches a bad
#       reset / merge that dropped the directory from the branch tip);
#   (b) the canary file exists on disk (catches a bad checkout of a commit
#       that has _docs/ but where the working tree somehow lost it).
TREE_HAS_CANARY=0
if git ls-tree --name-only HEAD -- "${CANARY}" 2>/dev/null | grep -q .; then
  TREE_HAS_CANARY=1
fi

DISK_HAS_CANARY=0
if [[ -f "${CANARY}" ]]; then
  DISK_HAS_CANARY=1
fi

if [[ "${TREE_HAS_CANARY}" == "1" && "${DISK_HAS_CANARY}" == "1" ]]; then
  exit 0
fi

# Something is wrong. Try to find a recoverable commit (in branch history or
# in the reflog) that still has the canary, so we can tell the user exactly
# what to run.
RECOVERY_COMMIT="$(git log --all --reflog --pretty=format:'%H' -1 -- "${CANARY}" 2>/dev/null || true)"

echo ""
echo "================================================================" >&2
echo "⚠️  _docs/ GUARDRAIL — drop detected on dev_xinli_2601" >&2
echo "================================================================" >&2
if [[ "${TREE_HAS_CANARY}" == "0" ]]; then
  echo "  • HEAD ($(git rev-parse --short HEAD)) does NOT contain:" >&2
  echo "      ${CANARY}" >&2
  echo "  • Most likely cause: a 'git reset --hard origin/main' (or merge" >&2
  echo "    that swept _docs/) while on dev_xinli_2601. This is the same" >&2
  echo "    regression that occurred on 2026-06-03 (reflog 4d4c76d) and" >&2
  echo "    was fixed by commit 9bddfd4." >&2
elif [[ "${DISK_HAS_CANARY}" == "0" ]]; then
  echo "  • HEAD contains ${CANARY} but the file is missing from disk." >&2
  echo "  • Run: git checkout -- _docs/" >&2
fi

if [[ -n "${RECOVERY_COMMIT}" ]]; then
  echo "" >&2
  echo "  Recovery: a known-good snapshot exists at ${RECOVERY_COMMIT:0:7}." >&2
  echo "  To restore:" >&2
  echo "      git checkout ${RECOVERY_COMMIT:0:7} -- _docs/" >&2
  echo "      git commit -m 'fix: restore _docs/ after accidental drop'" >&2
  echo "" >&2
  echo "  (See commit 9bddfd4 for the established pattern.)" >&2
else
  echo "" >&2
  echo "  No recoverable snapshot found in local refs/reflog." >&2
  echo "  Try: git fetch --all && rerun this check." >&2
fi
echo "================================================================" >&2

# Exit 1 so CI fails. The post-checkout hook will still allow the checkout
# (post-checkout's exit code doesn't block the checkout itself), it just
# prints the warning loudly so you notice immediately.
exit 1
