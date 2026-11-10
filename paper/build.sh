#!/usr/bin/env bash
# Rebuild both PDFs from the current .tex and .bib.
set -euo pipefail
cd "$(dirname "$0")"

TECTONIC="${TECTONIC:-$(command -v tectonic || true)}"
if [ -z "${TECTONIC}" ]; then
  echo "tectonic not found. Install it (https://tectonic-typesetting.github.io) or set TECTONIC=/path/to/tectonic." >&2
  exit 1
fi

"${TECTONIC}" -X compile neurips_3_method.tex
"${TECTONIC}" -X compile neurips_3_method_appendix.tex
echo "rebuilt neurips_3_method.pdf and neurips_3_method_appendix.pdf"
