# Diagrams (schematics) for the MetalFish papers

Hand-drawn schematics live here as self-contained HTML files with inline SVG, made with the
[diagram-design](https://github.com/cathrynlavery/diagram-design) skill (MIT). Plots of measured
data do not belong here: they come from `tools/paper/make_figures.py` and the measurement
artifacts. `make_figures.py` also exports every `*.html` here to a vector PDF in
`docs/paper_v2/figures/` (via `tools/paper/diagram_pdf.py`), which the manuscript includes.

| File | Figure |
| --- | --- |
| `hybrid-architecture.html` | Fig. 1, the hybrid runtime as frozen (architecture type) |

## Setup on a new machine

1. Install the skill as a Claude Code plugin (version 2.6.107 was reviewed and installed on
   2026-10-16):
   ```bash
   claude plugin marketplace add cathrynlavery/diagram-design
   claude plugin install diagram-design@diagram-design
   ```
2. Restore the print profile the project marker `.diagram-design` (`profile: metalfish-paper`)
   points to:
   ```bash
   mkdir -p ~/.diagram-design/profiles
   tail -n +4 docs/paper_v2/diagrams/metalfish-paper.profile.md > ~/.diagram-design/profiles/metalfish-paper.md
   ```
   The profile is for print: white paper, the system Times / Helvetica Neue / Menlo faces (no
   network fetch), one accent, and a 624-px canvas (6.5 in at 96 dpi, so 1 px prints as 0.75 pt
   at `\linewidth`) with a raised type ramp (node names 13 px, sublabels 10 px, labels 9 px).
3. Create the export environment (kept apart from the pinned measurement environment):
   ```bash
   /opt/homebrew/opt/python@3.14/bin/python3.14 -m venv ~/.venvs/diagram-design
   ~/.venvs/diagram-design/bin/pip install playwright
   ~/.venvs/diagram-design/bin/python -m playwright install chromium-headless-shell
   ```

## Checks before a diagram enters the paper

```bash
SK=~/.claude/plugins/cache/diagram-design/diagram-design/<version>/skills/diagram-design
~/.venvs/diagram-design/bin/python "$SK/scripts/self_check.py" --offline docs/paper_v2/diagrams/<file>.html
~/.venvs/diagram-design/bin/python tools/paper/diagram_pdf.py docs/paper_v2/diagrams/<file>.html --out docs/paper_v2/figures
```

The skill's connector-geometry verifier (`scripts/verify-geometry.py`) ships only in the
repository checkout, not in the installed plugin; run it from a clone of the reviewed commit.
