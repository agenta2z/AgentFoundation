#!/usr/bin/env bash
# restore.sh — replay a usecase: for each repo entry in repos.yaml, fetch
# from origin and checkout the pinned commit (detached HEAD).
#
# Generic across all usecases. Read-only on YAML (never edits). Refuses to
# clobber a dirty working tree without --force.
#
# Usage:
#   bash scripts/restore.sh [--usecase-dir <dir>] [--force] [--dry-run]
#
# Exit codes:
#   0  all repos restored
#   1  one or more repos failed (commit unreachable, dirty tree without --force, etc.)
#   2  YAML parse error / missing repos.yaml

set -euo pipefail

err() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARN:  $*" >&2; }
info() { echo "INFO:  $*" >&2; }

USECASE_DIR=""
FORCE=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --usecase-dir) USECASE_DIR="$2"; shift 2 ;;
    --force)       FORCE=1; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '2,14p' "$0"; exit 0 ;;
    *) err "unexpected arg: $1" ;;
  esac
done

if [[ -z "$USECASE_DIR" ]]; then
  USECASE_DIR="$(cd "$(dirname "$0")/.." && pwd -P)"
fi
REPOS_YAML="$USECASE_DIR/repos.yaml"
[[ -f "$REPOS_YAML" ]] || { echo "ERROR: $REPOS_YAML not found" >&2; exit 2; }

# ---------- parse repos.yaml ----------
# Minimal dependency-free YAML parser: extract triplets of (path, origin, commit)
# under each '- path:' entry. Brittle to YAML edits — keep schema flat.
python3 - "$REPOS_YAML" <<'PY' > "$USECASE_DIR/.restore.tsv"
import sys, re
path = sys.argv[1]
with open(path) as f: txt = f.read()
# Split on top-level "  - path:" entries
blocks = re.split(r'\n  - path:\s*', '\n' + txt)
for b in blocks[1:]:
    p = b.splitlines()[0].strip()
    def grab(key):
        m = re.search(r'^\s{4}' + re.escape(key) + r':\s*(.+?)\s*$', b, re.M)
        return m.group(1).strip() if m else ''
    origin = grab('origin')
    commit = grab('commit')
    branch = grab('branch')
    print('\t'.join([p, origin, commit, branch]))
PY

ANY_FAIL=0
while IFS=$'\t' read -r REPO ORIGIN COMMIT BRANCH; do
  [[ -z "$REPO" ]] && continue
  info "--- $REPO @ $COMMIT (was on $BRANCH) ---"

  if [[ ! -d "$REPO/.git" ]]; then
    warn "repo dir missing or not a git repo: $REPO"
    warn "  HINT: clone first:  git clone $ORIGIN $REPO"
    ANY_FAIL=1; continue
  fi

  cd "$REPO"

  DIRTY="$(git status --porcelain | wc -l | tr -d ' ')"
  if [[ "$DIRTY" -gt 0 && "$FORCE" -eq 0 ]]; then
    warn "working tree is dirty ($DIRTY entries); refusing to checkout."
    warn "  Use --force to override, or stash/commit first."
    ANY_FAIL=1; continue
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    info "(dry-run) would run: git fetch && git checkout $COMMIT"
    continue
  fi

  info "fetching from origin..."
  if ! git fetch --quiet "$ORIGIN" 2>/dev/null && ! git fetch --quiet origin 2>/dev/null; then
    warn "git fetch failed; will try checkout anyway in case commit is local"
  fi

  if ! git rev-parse --verify --quiet "$COMMIT^{commit}" >/dev/null; then
    warn "commit $COMMIT not found locally or on origin — was it force-pushed away?"
    ANY_FAIL=1; continue
  fi

  info "checking out $COMMIT (detached HEAD)..."
  if [[ "$FORCE" -eq 1 ]]; then
    git checkout --force --detach "$COMMIT"
  else
    git checkout --detach "$COMMIT"
  fi

  info "OK — $REPO is now at $(git rev-parse HEAD)"
done < "$USECASE_DIR/.restore.tsv"

rm -f "$USECASE_DIR/.restore.tsv"

if [[ "$ANY_FAIL" -ne 0 ]]; then
  echo ""
  warn "one or more repos failed to restore; see WARN lines above."
  exit 1
fi

echo ""
info "all repos restored. NOTE: plans/ directory is the canonical eval input — verify md5s in plan_sources.yaml match plans/*.md before judging."
echo "OK"
