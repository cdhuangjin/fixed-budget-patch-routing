# Experiment 027 — Author Actions Before Submission

All experiment, canonicalization, figure/table, manuscript, and audit work is
complete and verified for the Signal, Image and Video Processing (SIViP)
submission package. The items below are author- or CI-controlled and are the
only remaining items in `027_SIViP_论文拒稿后重构执行方案.md`.

## 1. Public branch — completed

The public repository is:

```text
https://github.com/cdhuangjin/fixed-budget-patch-routing
```

The submission-matched snapshot is maintained on:

```text
https://github.com/cdhuangjin/fixed-budget-patch-routing/tree/sivip_rebuild
```

The branch contains the refactored code, selected canonical derived results,
submission sources, compiled PDFs, and the self-contained source archive.

## 2. Record the submission-matched commit hash

Record the commit used for the submission. From the local project root:

```powershell
git rev-parse HEAD
```

Record the public hash in the submission record. The manuscript already links
to the public repository; do not replace that link with a private local path.

## 3. Optional archival release

For stronger long-term provenance, optionally create a signed release tag and
archive the same commit in a DOI-issuing repository. Do not point the submission
record at a different snapshot.

## 4. Submission-system metadata (author-supplied)

In the journal portal, confirm:

- Author names, affiliations, and email addresses.
- Funding statement and funding source details.
- Competing-interest / conflict-of-interest statement.
- Author contributions statement.
- Dataset licenses and benchmark versions permit redistribution of code and
  derived source data.
- Routing route: subscription (non-OA/no-APC) as intended.

## 5. Upload the package

Upload the editable sources and compiled PDFs from
`07_paper/submission/sivip_submission/`:

```text
SIViP_Experiment027_Manuscript_LaTeX_20260829.zip
```

This package is self-contained: it compiles `main.tex`, `supplementary.tex`, and
`cover_letter.tex` with Tectonic from a fresh extraction.

Use `build/main.pdf` for the manuscript, `build/ESM_1.pdf` for supplementary
material, and `build/cover_letter.pdf` for the cover-letter slot. Enter the
author-contribution and competing-interest statements again in the submission
system, as required by the journal workflow.

## Claim boundary (already verified in the manuscript)

The paper claims improved local-patch allocation quality under a fixed budget.
It does not claim end-to-end acceleration, universal online robustness, or
superiority over full-budget detectors.
