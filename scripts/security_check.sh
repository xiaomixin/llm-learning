#!/usr/bin/env bash
# Security check: scan staged (or entire repo) for secrets before commit/push.
# Exit 0 if clean, non-zero if suspicious content found.
#
# Usage:
#   scripts/security_check.sh                 # scan staged files (pre-commit mode)
#   scripts/security_check.sh --all           # scan ALL tracked + new files in repo
#   scripts/security_check.sh --files F1 F2   # scan specified files only

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:---staged}"
FILES=()

case "$MODE" in
  --staged)
    # Files staged for commit (added/modified, not deleted)
    while IFS= read -r f; do FILES+=("$f"); done < <(git diff --cached --name-only --diff-filter=ACM 2>/dev/null)
    ;;
  --all)
    while IFS= read -r f; do FILES+=("$f"); done < <(git ls-files -co --exclude-standard 2>/dev/null)
    ;;
  --files)
    shift
    FILES=("$@")
    ;;
  *)
    echo "Usage: $0 [--staged|--all|--files F1 F2 ...]" >&2
    exit 2
    ;;
esac

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "[security_check] no files to scan"
  exit 0
fi

# ── Scan rules ────────────────────────────────────────────────────────────────
# Each rule: "LABEL|REGEX"
# Use extended regex (grep -E). Keep patterns tight to avoid false positives.

declare -a RULES=(
  # Secrets / tokens
  "GitHub PAT (ghp_)|ghp_[A-Za-z0-9]{36}"
  "GitHub OAuth (gho_)|gho_[A-Za-z0-9]{36}"
  "GitHub fine-grained PAT|github_pat_[A-Za-z0-9_]{82}"
  "Generic bearer|Bearer [A-Za-z0-9_\\-\\.]{20,}"
  "AWS access key|AKIA[0-9A-Z]{16}"
  "Slack token|xox[baprs]-[0-9a-zA-Z]{10,}"
  "Google API key|AIza[0-9A-Za-z_\\-]{35}"
  "Private key PEM|-----BEGIN (RSA |EC |DSA |OPENSSH |)PRIVATE KEY-----"
  "OpenAI key|sk-[A-Za-z0-9]{32,}"
  "Anthropic key|sk-ant-[A-Za-z0-9_\\-]{20,}"
  "JWT|eyJ[A-Za-z0-9_\\-]{10,}\\.eyJ[A-Za-z0-9_\\-]{10,}\\.[A-Za-z0-9_\\-]{10,}"
  "Kaggle key JSON|\"key\"[[:space:]]*:[[:space:]]*\"[0-9a-f]{32}\""
  "Password assignment|(password|passwd|pwd)[[:space:]]*=[[:space:]]*['\\\"][^'\\\"]{6,}['\\\"]"

  # Kaggle credentials file contents
  "kaggle username pattern|\"username\"[[:space:]]*:[[:space:]]*\"[^\"]+\""
)

# Allowlist file — each line a literal string that, if present in the hit line, marks it safe.
ALLOWLIST_FILE="$REPO_ROOT/.security-allowlist"

# ── Exclude binary / generated ────────────────────────────────────────────────
EXCLUDED_SUFFIXES=(.png .jpg .jpeg .gif .pdf .zip .gz .tar .bin .model .pt .pth .onnx .parquet .csv)

is_excluded() {
  local f="$1"
  for suf in "${EXCLUDED_SUFFIXES[@]}"; do
    [[ "$f" == *"$suf" ]] && return 0
  done
  return 1
}

# ── Run ──────────────────────────────────────────────────────────────────────
HITS=0
HITS_LOG="/tmp/transformer-learning/security_check_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$HITS_LOG")"
: > "$HITS_LOG"

echo "[security_check] scanning ${#FILES[@]} file(s) with ${#RULES[@]} rules..."

for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || continue
  is_excluded "$f" && continue

  for rule in "${RULES[@]}"; do
    label="${rule%%|*}"
    pattern="${rule#*|}"

    matches=$(grep -nHE "$pattern" "$f" 2>/dev/null || true)
    [[ -z "$matches" ]] && continue

    # Apply allowlist
    if [[ -f "$ALLOWLIST_FILE" ]]; then
      filtered=""
      while IFS= read -r line; do
        safe=0
        while IFS= read -r token; do
          [[ -z "$token" || "$token" == \#* ]] && continue
          if grep -qF "$token" <<<"$line"; then safe=1; break; fi
        done < "$ALLOWLIST_FILE"
        [[ $safe -eq 0 ]] && filtered+="$line"$'\n'
      done <<< "$matches"
      matches="$filtered"
    fi

    [[ -z "${matches//[[:space:]]/}" ]] && continue

    HITS=$((HITS + 1))
    {
      echo "---"
      echo "[HIT] $label"
      echo "$matches"
    } | tee -a "$HITS_LOG"
  done
done

echo ""
if [[ $HITS -gt 0 ]]; then
  echo "❌ [security_check] $HITS issue(s) found. Details: $HITS_LOG"
  echo ""
  echo "To override (only if you're SURE it's a false positive):"
  echo "  1. Add the safe token to $REPO_ROOT/.security-allowlist (one per line)"
  echo "  2. Or bypass this hook just this once: git commit --no-verify"
  exit 1
else
  echo "✅ [security_check] no secrets found"
  exit 0
fi
