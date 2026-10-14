#!/bin/bash
# Build the manuscript: bibliography from the verified reference map (cached fetches), figures and
# tables from the measurement artifacts, then latexmk.
#   --final  every \pending{} becomes a compile error (arXiv-ready check)
#   --anon   double-anonymous variant for IEEE ToG -> main_anon.pdf
set -euo pipefail
cd "$(dirname "$0")"
python3 ../../../tools/paper/make_bib.py --offline
python3 ../../../tools/paper/make_figures.py
JOB=main; PRE=""
for arg in "$@"; do
  case "$arg" in
    --final) PRE="${PRE}\\def\\FINAL{1}" ;;
    --anon)  PRE="${PRE}\\def\\ANON{1}"; JOB=main_anon ;;
    *) echo "unknown option $arg"; exit 2 ;;
  esac
done
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname="$JOB" -usepretex="$PRE" main.tex
echo "built $JOB.pdf ($(pdfinfo "$JOB.pdf" | awk '/Pages/{print $2}') pages)"
