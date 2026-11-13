#!/usr/bin/env bash
# Rebuild the paper PDF from SBchallenge26_round1.tex.
set -euo pipefail
cd "$(dirname "$0")"

TECTONIC="${TECTONIC:-$(command -v tectonic || true)}"
if [ -z "${TECTONIC}" ]; then
  echo "tectonic not found. Install it (https://tectonic-typesetting.github.io) or set TECTONIC=/path/to/tectonic." >&2
  exit 1
fi

TEX="${1:-SBchallenge26_round1.tex}"
"${TECTONIC}" -X compile "${TEX}"
echo "rebuilt ${TEX%.tex}.pdf"
