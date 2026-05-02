#!/usr/bin/env bash
# Sync notebooks + light artifacts from this repo to Google Drive,
# so Colab can open them from MyDrive (internal git isn't reachable from Colab).
#
# Usage:
#   scripts/sync_to_drive.sh             # local -> Drive (push)
#   scripts/sync_to_drive.sh pull        # Drive -> local (pull edits made in Colab)
#   scripts/sync_to_drive.sh diff        # show what would change
#
# Requires: DRIVE_DIR defined in scripts/sync.config.local (gitignored)
#   DRIVE_DIR="$HOME/Library/CloudStorage/GoogleDrive-<email>/My Drive/<your-folder>"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="$REPO_ROOT/scripts/sync.config.local"

if [[ ! -f "$CFG" ]]; then
  echo "ERROR: $CFG not found. Create it with a single line:" >&2
  echo "  DRIVE_DIR=\"\$HOME/Library/CloudStorage/GoogleDrive-<email>/My Drive/<folder>\"" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CFG"

: "${DRIVE_DIR:?DRIVE_DIR not set in $CFG}"

if [[ ! -d "$DRIVE_DIR" ]]; then
  echo "ERROR: DRIVE_DIR does not exist: $DRIVE_DIR" >&2
  echo "Create the folder inside Google Drive (desktop app must be running)." >&2
  exit 1
fi

MODE="${1:-push}"

# Files/folders to sync (notebooks, light scripts, docs). Exclude data/models.
INCLUDE_FLAGS=(
  --include='*/'               # descend into all dirs first
  --include='*.ipynb'
  --include='*.py'
  --include='*.md'
  --include='requirements.txt'
  --exclude='.git/***'
  --exclude='.githooks/***'
  --exclude='data/***'
  --exclude='models/***'
  --exclude='checkpoints/***'
  --exclude='__pycache__/***'
  --exclude='.ipynb_checkpoints/***'
  --exclude='.DS_Store'
  --exclude='*.log'
  --exclude='scripts/*.config.local'
  --exclude='*'                # anything else excluded
)

push() {
  echo "Push: $REPO_ROOT/  →  $DRIVE_DIR/"
  rsync -av "${INCLUDE_FLAGS[@]}" "$REPO_ROOT/" "$DRIVE_DIR/"
  echo "✅ Done. Drive desktop app will sync to cloud in seconds."
}

pull() {
  echo "Pull: $DRIVE_DIR/  →  $REPO_ROOT/"
  echo "⚠️  This overwrites local files. Ctrl-C within 3s to abort."
  sleep 3
  rsync -av "${INCLUDE_FLAGS[@]}" "$DRIVE_DIR/" "$REPO_ROOT/"
  echo "✅ Done. Review with: git status"
}

diff_() {
  echo "Dry-run push: $REPO_ROOT/  →  $DRIVE_DIR/"
  rsync -avn "${INCLUDE_FLAGS[@]}" "$REPO_ROOT/" "$DRIVE_DIR/" | sed -n '/^[^ ]/p' | head -40
}

case "$MODE" in
  push) push ;;
  pull) pull ;;
  diff) diff_ ;;
  *) echo "Usage: $0 [push|pull|diff]" >&2; exit 2 ;;
esac
