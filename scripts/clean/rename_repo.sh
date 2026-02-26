#!/usr/bin/env bash

# Rename project occurrences and directories.
# Usage:
#   ./scripts/rename_repo.sh                  # default: GopsEA -> GopsEA
#   ./scripts/rename_repo.sh OldName NewName  # custom names
#
# It does two things:
# 1) Replace all text occurrences of OLD_NAME with NEW_NAME in files under repo root.
# 2) Rename all directories whose *name* exactly equals OLD_NAME to NEW_NAME.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OLD_NAME="${1:-GopsEA}"
NEW_NAME="${2:-GopsEA}"

if [[ "$OLD_NAME" == "$NEW_NAME" ]]; then
  echo "OLD_NAME and NEW_NAME are the same: '$OLD_NAME'. Nothing to do."
  exit 0
fi

echo "Repository root: $REPO_ROOT"
echo "Replacing text: '$OLD_NAME' -> '$NEW_NAME'"
echo "Renaming directories named '$OLD_NAME' -> '$NEW_NAME'"
echo

read -r -p "Continue? [y/N]: " CONFIRM || true
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 1
fi

cd "$REPO_ROOT"

echo "Step 1/2: Replacing text in files..."

# Find regular files (excluding .git) and replace text.
find "$REPO_ROOT" \
  -type f \
  -not -path "*/.git/*" \
  -print0 \
  | xargs -0 sed -i "s/${OLD_NAME}/${NEW_NAME}/g"

echo "Step 2/2: Renaming directories..."

# Rename directories whose name exactly matches OLD_NAME, bottom‑up.
find "$REPO_ROOT" \
  -depth -type d -name "$OLD_NAME" \
  -print0 \
  | while IFS= read -r -d '' DIR; do
      PARENT="$(dirname "$DIR")"
      BASENAME="$(basename "$DIR")"
      if [[ "$BASENAME" == "$OLD_NAME" ]]; then
        TARGET="$PARENT/$NEW_NAME"
        echo "Renaming directory: $DIR -> $TARGET"
        mv "$DIR" "$TARGET"
      fi
    done

echo "Done."
echo "Old name: $OLD_NAME"
echo "New name: $NEW_NAME"

