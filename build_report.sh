#!/bin/bash
# Compile the Bessembinder report, refresh page renders, open the PDF.
cd "$(dirname "$0")" || exit 1
pdflatex -interaction=nonstopmode -halt-on-error bessembinder_corrections.tex
status=$?
rm -f bessembinder_corrections.aux bessembinder_corrections.log
rm -f /tmp/bess_pg*.png
pdftoppm -png -r 110 bessembinder_corrections.pdf /tmp/bess_pg
open bessembinder_corrections.pdf
exit $status
