#!/bin/bash
# Build the kickoff deck: figures, then two pdfLaTeX passes. Run from anywhere.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
mkdir -p "$here/logos"
if [ "${1:-}" != "--no-figures" ]; then
  (cd "$root/code/scripts/deck" && CUDA_VISIBLE_DEVICES="" "$root/.venv/bin/python" run_all.py | tail -1)
fi
cd "$here"
pdflatex -interaction=nonstopmode -halt-on-error kickoff_deck.tex > build.log
pdflatex -interaction=nonstopmode -halt-on-error kickoff_deck.tex >> build.log
rm -f kickoff_deck.aux kickoff_deck.log kickoff_deck.nav kickoff_deck.out kickoff_deck.snm kickoff_deck.toc build.log
echo "wrote $here/kickoff_deck.pdf"
