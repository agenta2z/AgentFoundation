#!/usr/bin/env bash
# scripts/install-git-hooks.sh
#
# Installs the project's recommended git hooks into .git/hooks/.
# Idempotent — safe to re-run. Preserves any existing hooks by chaining
# (we append a sentinel-fenced block, not overwrite blindly).
#
# Hooks installed:
#   post-checkout  → runs scripts/check_dev_docs_present.sh
#
# Usage:
#   bash scripts/install-git-hooks.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

HOOKS_DIR=".git/hooks"
mkdir -p "${HOOKS_DIR}"

SENTINEL_BEGIN="# >>> agent_foundation hook block >>>"
SENTINEL_END="# <<< agent_foundation hook block <<<"

install_hook() {
  local hook_name="$1"
  local hook_body="$2"
  local hook_path="${HOOKS_DIR}/${hook_name}"

  # Create empty hook with shebang if it doesn't exist
  if [[ ! -f "${hook_path}" ]]; then
    printf '#!/usr/bin/env bash\nset -e\n' > "${hook_path}"
  fi

  # If our block already exists, replace it; otherwise append.
  if grep -qF "${SENTINEL_BEGIN}" "${hook_path}"; then
    # Replace the existing block in-place
    awk -v begin="${SENTINEL_BEGIN}" -v end="${SENTINEL_END}" -v body="${hook_body}" '
      BEGIN { skip = 0 }
      $0 == begin { print begin; print body; print end; skip = 1; next }
      $0 == end   { skip = 0; next }
      skip == 0   { print }
    ' "${hook_path}" > "${hook_path}.tmp"
    mv "${hook_path}.tmp" "${hook_path}"
  else
    {
      echo ""
      echo "${SENTINEL_BEGIN}"
      echo "${hook_body}"
      echo "${SENTINEL_END}"
    } >> "${hook_path}"
  fi

  chmod +x "${hook_path}"
  echo "✓ installed ${hook_name}"
}

POST_CHECKOUT_BODY='# Guardrail: ensure dev_xinli_2601 still has _docs/ tree
# (see scripts/check_dev_docs_present.sh for context)
bash "$(git rev-parse --show-toplevel)/scripts/check_dev_docs_present.sh" || true'

install_hook "post-checkout" "${POST_CHECKOUT_BODY}"

# Also install scripts/check_dev_docs_present.sh as executable
chmod +x "${REPO_ROOT}/scripts/check_dev_docs_present.sh" 2>/dev/null || true

echo ""
echo "Git hooks installed. To verify, run:"
echo "    bash scripts/check_dev_docs_present.sh && echo OK"
