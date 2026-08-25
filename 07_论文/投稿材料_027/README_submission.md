# Experiment 027 — PAAA submission package

Target journal: [Pattern Analysis and Applications](https://link.springer.com/journal/10044). The intended route is subscription publication; the journal's official publishing page describes the subscription route as having no article processing charge.

## Files

- `main.tex`: Springer Nature manuscript source.
- `build/main.pdf`: compiled manuscript.
- `supplementary.tex`: supplementary information source.
- `build/supplementary.pdf`: compiled supplementary information.
- `cover_letter.tex`: cover letter source.
- `figure_source_data.csv`: source values for Figures 1–3.
- `plot_paaa_figures.py`: figure regeneration script.
- `figures/`: vector PDF, SVG, 600-dpi TIFF, and PNG figure exports.

## Literature synchronization

The manuscript bibliography is synchronized across `main.tex`, `references.bib`, `build/main.bbl`, and `build/main.pdf`. It contains 19 cited works, including recent 2023--2025 methods and surveys and the 2026 ISP-AD real-world benchmark. The supplementary information and cover letter do not contain independent literature claims, so they do not maintain a second bibliography.

## Submission checks still requiring author confirmation

1. Replace the repository placeholder with the final public repository URL and commit identifier.
2. Confirm author names, affiliations, emails, funding statement, competing-interest statement, and contribution statement in the submission system.
3. Upload editable `main.tex`, bibliography, class/style files, figure source files, and supplementary source as required by the journal portal.
4. Upload the compiled PDF for local review only if the portal requests it; use the portal's source-file compilation workflow where applicable.
5. Confirm that all dataset licenses and downloaded benchmark versions permit the planned redistribution of code and derived source data.
6. Select the subscription route if the goal remains non-OA/no-APC publication.

## Claim boundary

The paper claims improved local-patch allocation quality under a fixed computational budget. It does not claim end-to-end acceleration, universal online robustness, or superiority over full-budget anomaly detectors.
