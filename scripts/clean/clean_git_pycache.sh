#!/usr/bin/env bash
set -euo pipefail

# Remove the following from all subdirectories of this project:
#   - .git directories (excluding the repo root .git)
#   - __pycache__ directories
#   - *.egg-info directories/files
#
# Usage (from repo root or any directory):
#   bash scripts/clean_git_pycache.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Repo root: ${REPO_ROOT}"

read -r -p "Are you sure you want to delete subdirectory .git / __pycache__ / *.egg-info ? (y/N) " CONFIRM
if [[ ! "${CONFIRM:-}" =~ ^[Yy]$ ]]; then
  echo "Operation cancelled."
  exit 0
fi

echo "Cleaning..."

# 1. Remove all .git directories except the repo root one
echo "Removing .git directories in subfolders (keeping repo root .git)..."
find "${REPO_ROOT}" \
  -path "${REPO_ROOT}/.git" -prune -o \
  -name ".git" -type d -print -exec rm -rf {} +

# 2. Remove all __pycache__ directories
echo "Removing __pycache__ directories..."
find "${REPO_ROOT}" -name "__pycache__" -type d -print -exec rm -rf {} +

# 3. Remove all *.egg-info
echo "Removing *.egg-info ..."
find "${REPO_ROOT}" -name "*.egg-info" -print -exec rm -rf {} +

echo "Done."

