# Experiment 027 — SIViP submission package

Target journal: [Signal, Image and Video Processing](https://link.springer.com/journal/11760).

This package is the SIViP-facing, editable submission source derived from the
verified `canonical_v3/` strong-routing evidence (99 category--seed rows,
seeds 5/17/29) and the MVTec `canonical_v3_budget/` sensitivity result. Its
figures and tables are generated from those canonical result directories; the
historical submission package remains separate at `../投稿材料_027/`.

## Contents

- `main.tex`, `references.bib`, `sn-jnl.cls`, and `sn-basic.bst`:
  editable manuscript sources using the journal-recommended two-column Basic
  Springer Nature reference style.
- `supplementary.tex` and `cover_letter.tex`: editable accompanying sources.
- `build/main.pdf`, `build/ESM_1.pdf`, and `build/cover_letter.pdf`: compiled
  files ready for the manuscript, supplementary-material, and cover-letter
  upload slots. `build/supplementary.pdf` is retained as a source-name alias.
- `figures/`: canonical V3 routing, V2 latency-boundary, and budget-sensitivity
  figures required by the manuscript.
- `data/canonical_v3/` and `data/canonical_v3_budget/`: the derived
  category--seed tables, display tables, manifests, and audit reports that
  support Table 1, Figures 1--2, and the supplementary tables. Benchmark
  images are not redistributed.
- `data/upstream/`: the two compact, checksum-pinned canonical JSON records
  from which the public CSV tables were materialized.
- `reference_verification_report.md`: citation verification record.
- `AUTHOR_ACTIONS_027.md`: the remaining author/CI-controlled release steps.

## Reproducibility status

The submission-matched source, derived results, figures, and compiled artifacts
are published on the public `sivip_rebuild` branch. Use the Git commit containing
this README as the immutable revision for the submission record. A DOI-backed
archive or release tag has not yet been assigned. Do not cite the historical
initial release as the source for this refactor.

## Local build

Compile `main.tex`, `supplementary.tex`, and `cover_letter.tex` with the bundled
Tectonic executable. The main manuscript uses
`\documentclass[iicol,pdflatex,sn-basic,Numbered]{sn-jnl}` and compiles to nine
A4 pages; the final page contains references only. Build from this directory
so the relative figure paths resolve as written.
