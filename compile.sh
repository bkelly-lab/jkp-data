#!/bin/bash
# Usage: bash compile.sh <basename-without-.tex>
set -e
b="$1"
pdflatex -interaction=nonstopmode -halt-on-error "$b.tex" >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error "$b.tex" >/dev/null
rm -f "$b.aux" "$b.log" "$b.out"
open "$b.pdf"
echo "compiled $b.pdf"
