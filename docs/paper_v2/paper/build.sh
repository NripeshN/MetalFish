#!/bin/bash
# Build the manuscript: bibliography from the verified reference map (cached fetches), figures and
# tables from the measurement artifacts, then latexmk.
#   --final  every \pending{} becomes a compile error (arXiv-ready check)
#   --anon   double-anonymous variant for IEEE ToG -> main_anon.pdf
set -euo pipefail
cd "$(dirname "$0")"
# The measurement venv pins the interpreter and packages (the host's default python3 may lack them).
PY=${PY:-../../../results/paper_v2/venv/bin/python}
[ -x "$PY" ] || PY=python3
JOB=main; PRE=""; FIG=""
for arg in "$@"; do
  case "$arg" in
    --final) PRE="${PRE}\\def\\FINAL{1}"; FIG="--final-analyses" ;;
    --anon)  PRE="${PRE}\\def\\ANON{1}"; JOB=main_anon ;;
    *) echo "unknown option $arg"; exit 2 ;;
  esac
done
"$PY" ../../../tools/paper/make_bib.py --offline
# --final also writes the final-reading analyses (amendment 12; slow)
"$PY" ../../../tools/paper/make_figures.py $FIG
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname="$JOB" -usepretex="$PRE" main.tex
echo "built $JOB.pdf ($(pdfinfo "$JOB.pdf" | awk '/Pages/{print $2}') pages)"
