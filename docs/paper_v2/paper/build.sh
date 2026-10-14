#!/bin/bash
# Build the manuscript: bibliography from the verified reference map, figures/tables from the
# measurement artifacts, then latexmk. Pass --final to turn every \pending{} into a compile error.
set -euo pipefail
cd "$(dirname "$0")"
python3 ../../../tools/paper/make_bib.py --offline
python3 ../../../tools/paper/make_figures.py
if [ "${1:-}" = "--final" ]; then
  sed -i.bak 's/^\\finalfalse$/\\finaltrue/' main.tex && rm -f main.tex.bak
fi
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
echo "built main.pdf ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages)"
