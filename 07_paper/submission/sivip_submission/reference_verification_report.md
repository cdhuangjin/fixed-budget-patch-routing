# Experiment 027 — citation and reference verification

Verification date: 2026-08-25
Target journal: Signal, Image and Video Processing (SIViP)
Scope: full `references.bib`, 20 entries; Crossref DOI checks plus official CVF/Springer/Journal and dataset records where available.

## Result summary

- Verified or corrected: 20/20 entries are now usable in the manuscript.
- Critical correction applied: PatchCore DOI changed from `10.1109/CVPR52688.2022.01393` to `10.1109/CVPR52688.2022.01392`; the former resolves to H2FA R-CNN, whereas the latter resolves to Roth et al.'s PatchCore paper.
- Critical metadata correction applied: the 2024 industrial-anomaly survey now uses the Crossref author list for Liu et al., not the previously assigned Tao et al. authors.
- Field correction applied: the 2025 industrial visual anomaly survey now records volume 58, issue 9, article 279.
- Normalized a malformed author token in the 2024 tiled-ensemble entry.
- Added the primary MPDD dataset paper (Jezek et al., ICUMT 2021; DOI
  `10.1109/ICUMT54235.2021.9631567`).
- Added verified IEEE DOIs for eight CVPR/WACV entries that previously lacked
  DOI metadata.
- Completed the ISP-AD record with Springer DOI
  `10.1007/s10845-025-02778-z`; the article was published online on
  31 January 2026 and does not yet have volume/page metadata in Crossref.

## Remaining source-level warnings

1. MVTec AD and PatchCore have a page-range discrepancy between Crossref and the official CVF landing pages. The manuscript retains the official CVF page ranges: MVTec AD 9592--9600 and PatchCore 14318--14328. This is flagged for editorial verification, not treated as a fabricated citation.
2. Geifman and El-Yaniv's selective-classification paper has no DOI in the current record. It remains a standard NeurIPS 2017 reference and should be checked against the final publisher bibliography if the journal portal requires DOI completion.
3. Vovk et al. is a book; Crossref returns incomplete personal-author fields, so the author list is retained from the publisher/book record rather than from the incomplete Crossref response.

## Citation-support audit

The manuscript's current citations are appropriate for the claims they accompany:

| Claim area | Current support | Assessment |
|---|---|---|
| MVTec AD and VisA benchmark context | Bergmann et al. 2019; Zou et al. 2022 | Strong/background support |
| Patch memory and coreset reference | Defard et al. 2021; Roth et al. 2022 | Strong method support |
| Recent efficiency and high-resolution directions | Liu et al. 2023; Batzner et al. 2024; Rolih et al. 2024 | Strong/partial method support |
| Vision-language and prompt-based IAD | Jeong et al. 2023; Li et al. 2024; Costanzino et al. 2024 | Background/partial support; not claimed as the proposed method |
| Industrial anomaly-detection field landscape | Liu et al. 2024; Li et al. 2025; Mao et al. 2025 | Strong review/context support |
| Real-world benchmark limitation | Krassnig and Gruber 2026 | Strong limiting/context support |
| Selective prediction and calibration | Geifman and El-Yaniv 2017; Vovk et al. 2005; Angelopoulos and Bates 2023 | Background/method support |

## Nature-family search outcome

The `nature-citation` run used five claim segments, 2020--2026, Nature Portfolio scope, and generated review artifacts in `audit_nature_citation_2026/`. It returned no direct Nature-family candidate for the recent industrial-methods segment. Several metadata-only hits were unrelated to industrial anomaly routing and were rejected. They should not be inserted merely to increase the Nature-family count.

## Output artifacts

- [Nature citation browser](audit_nature_citation_2026/references.html)
- [Nature citation report](audit_nature_citation_2026/references.md)
- [Nature citation RIS](audit_nature_citation_2026/references.ris)
- [Current BibTeX](references.bib)
