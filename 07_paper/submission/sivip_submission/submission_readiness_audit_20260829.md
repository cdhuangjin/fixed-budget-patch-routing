# Experiment 027 — SIViP submission-readiness audit (2026-08-29)

Scope: template, layout, figure-accessibility, reference-metadata and
submission-package compliance for Signal, Image and Video Processing. The
experimental data, statistical estimates and research conclusions were not
changed.

## Template and length — PASS

- The manuscript uses the official December 2024 `sn-jnl.cls`; its SHA-256
  matches the author-supplied template package.
- The main source uses
  `\documentclass[iicol,pdflatex,sn-basic,Numbered]{sn-jnl}`.
- `sn-basic.bst` was copied from the same official package and its SHA-256
  matches the template copy.
- The compiled main manuscript is nine A4 pages. Page 9 contains references
  only, satisfying the journal's ten-page final-format limit.
- Final logs contain no LaTeX errors, undefined citations/references or
  overfull boxes.

## Figure audit — PASS

- All three figures use double-column `figure*` floats and fit the two-column
  text width without clipping.
- Plot titles and caption-like text were removed from the illustrations; the
  corresponding explanations remain in the LaTeX captions.
- Legends do not overlap plotted data. Bar charts use hatches and outlines,
  and comparison points use distinct markers, so interpretation does not rely
  on colour alone.
- PDF, SVG, TIFF and PNG exports are synchronized in the analysis,
  paper-figure and submission directories.

## Front matter and declarations — PASS

- Abstract length remains within 150--250 words; AUROC, CUDA and P95 are
  expanded at first use.
- Five keywords are provided.
- The corresponding author's ORCID is linked without inserting a forced line
  break into the email field.
- The required heading is `Statements and Declarations`, followed by funding,
  competing interests, ethics/consent, data availability, code availability
  and author contributions.

## Data and code availability — PASS WITH RELEASE ACTION

- The statement provides direct public access links for MVTec AD, MPDD and
  VisA and cites their source publications.
- The public project repository is linked for derived data and code:
  `https://github.com/cdhuangjin/fixed-budget-patch-routing`.
- Before submission, the author should push and tag the exact
  submission-matched snapshot and record its immutable commit or release in
  the submission record. No private path appears in the manuscript.

## Citation and reference metadata — PASS

- The manuscript cites 20 BibTeX entries, with no missing or orphaned keys.
- The primary MPDD paper and DOI were added.
- Eight CVPR/WACV DOI records and the Springer ISP-AD DOI were verified and
  added. Geifman and El-Yaniv's NeurIPS 2017 paper remains the only entry
  without a DOI in the verified record.
- The numbered Basic Springer Nature style renders square-bracket citations
  and DOI links.

## Statistical reporting — REVIEWER RISK RECORDED

- The manuscript defines the category--seed unit, fixed-quota pairing and
  10,000 paired bootstrap resamples, and does not treat timing repetitions as
  independent accuracy units.
- A separate category-clustered sensitivity diagnostic is retained in
  `statistics_cluster_bootstrap_sensitivity_20260829.md` and JSON. The central
  Risk-versus-Random interval remains positive on MVTec AD, MPDD, VisA and the
  combined diagnostic. This compliance pass did not replace the preregistered
  manuscript estimates.

## Supplementary material and package — PASS

- The supplementary source names the article, journal, authors, affiliations
  and corresponding-author contact, and compiles to `build/ESM_1.pdf`.
- The editable source package contains the official class/style files,
  figures, bibliography, main manuscript, supplement, cover letter and
  compiled PDFs.

## Current gate

- Template/layout gate: PASS.
- Figure accessibility gate: PASS.
- Reference metadata gate: PASS.
- PDF compilation and visual inspection gate: PASS.
- Submission package gate: PASS after the archive is rebuilt.
- Public repository snapshot: PASS on the `sivip_rebuild` branch; record its
  commit hash in the submission log. A DOI-backed archive or release tag remains
  optional.
- Remaining author-controlled action: re-enter declarations and metadata in
  the submission portal.
