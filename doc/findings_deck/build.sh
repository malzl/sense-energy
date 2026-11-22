#!/bin/bash
# Build the findings deck: figures, then two pdfLaTeX passes. Run from anywhere.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
mkdir -p "$here/logos"
if [ "${1:-}" != "--no-figures" ]; then
  (cd "$root/code/scripts/findings_deck" && CUDA_VISIBLE_DEVICES="" "$root/.venv/bin/python" run_all.py | tail -1)
fi
cd "$here"
pdflatex -interaction=nonstopmode -halt-on-error findings_deck.tex > build.log
pdflatex -interaction=nonstopmode -halt-on-error findings_deck.tex >> build.log
rm -f findings_deck.aux findings_deck.log findings_deck.nav findings_deck.out findings_deck.snm findings_deck.toc build.log
echo "wrote $here/findings_deck.pdf"
