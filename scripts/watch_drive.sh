#!/usr/bin/env bash
# Watch the Google Drive folder for changes (Colab edits land there after save)
# and pull them back to the local repo. One-way: Drive → local.
#
# Usage:  scripts/watch_drive.sh
# Stop:   Ctrl-C
#
# Requires: fswatch  (brew install fswatch)
#           DRIVE_DIR set in scripts/sync.config.local

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="$REPO_ROOT/scripts/sync.config.local"

if ! command -v fswatch >/dev/null 2>&1; then
  echo "ERROR: fswatch not installed. Run: brew install fswatch" >&2
  exit 1
fi

if [[ ! -f "$CFG" ]]; then
  echo "ERROR: $CFG not found (needs DRIVE_DIR)." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CFG"
: "${DRIVE_DIR:?DRIVE_DIR not set}"

if [[ ! -d "$DRIVE_DIR" ]]; then
  echo "ERROR: DRIVE_DIR does not exist: $DRIVE_DIR" >&2
  exit 1
fi

# Debounce: Drive writes many small chunks; wait 2s after the last event.
echo "👀 Watching: $DRIVE_DIR"
echo "   → pulling .ipynb / .py / .md changes to $REPO_ROOT"
echo "   Ctrl-C to stop."

# --latency 2: coalesce bursts within 2s
# --event Updated --event Created --event Renamed: ignore pure metadata touches
fswatch --latency 2 \
        --event Updated --event Created --event Renamed \
        --exclude '\.tmp\.driveupload' \
        --exclude '\.DS_Store' \
        --exclude '\.ipynb_checkpoints' \
        "$DRIVE_DIR" | while read -r changed; do
  # Only act on files we care about
  case "$changed" in
    *.ipynb|*.py|*.md|*/requirements.txt)
      rel="${changed#"$DRIVE_DIR"/}"
      dst="$REPO_ROOT/$rel"
      mkdir -p "$(dirname "$dst")"
      # Only copy if content actually differs (rsync does a checksum compare)
      if rsync -c --quiet "$changed" "$dst"; then
        echo "[$(date +%H:%M:%S)] ⬇  $rel"
      fi
      ;;
  esac
done
