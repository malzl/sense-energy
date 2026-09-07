#!/usr/bin/env bash
# Apply the branch protection ruleset and merge settings for this repo.
#
# Run once, after `gh auth login`:
#     ./.github/setup-branch-protection.sh
#
# Uses a repository *ruleset* rather than classic branch protection, because
# classic protection on a private repo requires a paid plan while rulesets are
# available on Free. Re-running updates the existing ruleset in place.

set -euo pipefail

REPO="${1:-malzl/sense-energy}"
RULESET_NAME="protect-main"

command -v gh >/dev/null || { echo "gh not found on PATH"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Not authenticated. Run: gh auth login"; exit 1; }

echo "Configuring $REPO"

# --- Merge settings: squash-only, tidy up merged branches -------------------
gh api -X PATCH "repos/$REPO" \
  -F allow_squash_merge=true \
  -F allow_merge_commit=false \
  -F allow_rebase_merge=false \
  -F delete_branch_on_merge=true \
  -F allow_auto_merge=true \
  --silent
echo "  merge settings applied (squash only, auto-delete merged branches)"

# --- Branch ruleset ---------------------------------------------------------
# required_approving_review_count is 0 deliberately: with a single maintainer,
# requiring an approval makes main unmergeable. Raise it to 1 as soon as there
# is a second person on the project.
payload=$(cat <<'JSON'
{
  "name": "protect-main",
  "target": "branch",
  "enforcement": "active",
  "bypass_actors": [],
  "conditions": {
    "ref_name": { "include": ["refs/heads/main"], "exclude": [] }
  },
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 0,
        "dismiss_stale_reviews_on_push": true,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": false,
        "allowed_merge_methods": ["squash"]
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "do_not_enforce_on_create": false,
        "required_status_checks": [
          { "context": "test (3.11)" },
          { "context": "test (3.12)" },
          { "context": "no-data-committed" }
        ]
      }
    }
  ]
}
JSON
)

existing=$(gh api "repos/$REPO/rulesets" --jq \
  ".[] | select(.name==\"$RULESET_NAME\") | .id" 2>/dev/null || true)

if [ -n "$existing" ]; then
  echo "  updating existing ruleset $existing"
  printf '%s' "$payload" | gh api -X PUT "repos/$REPO/rulesets/$existing" --input - --silent
else
  echo "  creating ruleset"
  printf '%s' "$payload" | gh api -X POST "repos/$REPO/rulesets" --input - --silent
fi

echo
echo "main is now protected:"
echo "  - changes must arrive via pull request"
echo "  - CI (tests on 3.11/3.12 + the no-data guard) must pass, and the branch"
echo "    must be up to date with main before merging"
echo "  - no force pushes, no deletion"
echo "  - squash merges only; head branches deleted after merge"
