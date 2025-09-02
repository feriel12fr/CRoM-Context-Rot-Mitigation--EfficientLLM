#!/usr/bin/env bash
set -euo pipefail

TAG=${1:-}
if [[ -z "$TAG" ]]; then
  echo "Usage: scripts/release.sh vX.Y.Z"; exit 1
fi

# sanity checks
if [[ -n $(git status --porcelain) ]]; then
  echo "❌ Working tree not clean"; exit 1
fi

# ensure deps
python -m pip install -e .[dev]
pre-commit run --all-files
pytest -q

# generate release notes preview from CHANGELOG
python scripts/gen_release_notes.py "$TAG"
if [[ -f release_notes.md ]]; then
  echo "--- release_notes.md (preview top 60 lines) ---"
  head -n 60 release_notes.md || true
  echo "--- end preview ---"
else
  echo "⚠️ release_notes.md not generated; will fall back to default notes in GH release"
fi

# tag & push


git tag -a "$TAG" -m "Release $TAG"
git push origin "$TAG"

echo "✅ Pushed tag $TAG. GitHub Actions will create the Release automatically."
echo "➡️  Watch: https://github.com/Flamehaven/CRoM-EfficientLLM/actions"