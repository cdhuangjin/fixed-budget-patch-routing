# Experiment 027 — derived-data release manifest

Date: 2026-08-29
Scope: local pre-deposit inventory for the SIViP submission. This file does not
claim that a public repository or DOI already exists.

## Summary

This release inventory covers derived numeric results supporting the canonical
25% strong-routing comparison and the MVTec 10/25/50% budget-sensitivity
analysis. It does not include or redistribute MVTec AD, MPDD, or VisA benchmark
images. Users must obtain those datasets from their original providers under
the applicable dataset terms.

## Result sets

### Canonical 25% strong-routing result

- Protocol: exact fallback count `ceil(n_test * 0.25)` within every
  dataset-category-seed row.
- Seeds: 5, 17, 29.
- Rows: 99 (MVTec AD 45, MPDD 18, VisA 36).
- Routing policies: local-patch Risk, matched Random, Fast-score, and
  nearest-patch-distance dispersion Uncertainty.
- Primary source table: `05_运行记录/canonical_v3/raw_results.csv`.
- Derived display tables: `category_results.csv` and `main_results.csv`.
- Integrity audit: 4 PASS, 0 WARN, 0 FAIL.

### MVTec budget sensitivity

- Budgets: 10%, 25%, 50%.
- Seeds: 5, 17, 29.
- Rows: 135 (45 category--seed rows per budget).
- Primary source table: `05_运行记录/canonical_v3_budget/raw_results.csv`.
- Derived display table: `budget_sensitivity.csv`.
- Integrity audit: 4 PASS, 0 WARN, 0 FAIL.

## File checksums

SHA-256 values are lowercase hexadecimal.

| Relative file | Bytes | SHA-256 |
|---|---:|---|
| `05_运行记录/canonical_v3/raw_results.csv` | 25068 | `c53c45e371b2b97f368142fe5e689ee7da52627353dc4f12547de1f9bd398469` |
| `05_运行记录/canonical_v3/category_results.csv` | 3600 | `e37e76ca27bca44386982c7fa832628e74a78edf973d429b405bdc4de7e3d8ec` |
| `05_运行记录/canonical_v3/main_results.csv` | 1215 | `83fda88ea633533c53324b5d6b95b97bd2a83db5689f2d3f55e210e7bfe25f0b` |
| `05_运行记录/canonical_v3/manifest.json` | 791 | `494a6bfdc6afbfc499f764303723ff023311e4213eb322cfa0068b60f17336ad` |
| `05_运行记录/canonical_v3/audit_report.json` | 621 | `cd51e13abee51e3a75a9371c880d402b20e287ff053cda1711a4ab9d309767e7` |
| `05_运行记录/canonical_v3_budget/raw_results.csv` | 40189 | `1b31750bf934a68a19d7b589decf29b939e10aab742ded0fd1283d1fac8d65e0` |
| `05_运行记录/canonical_v3_budget/budget_sensitivity.csv` | 525 | `e96dd760358cc42ba03d348c33c8d79a0d8dcbb61ebfc38806a5e987d95b0a93` |
| `05_运行记录/canonical_v3_budget/manifest.json` | 883 | `d9dde9fd0c29c940f6a15625fd5afe42661dab80b615fd896a014ba9de80280a` |
| `05_运行记录/canonical_v3_budget/audit_report.json` | 681 | `e3f1d95e2dbc6d9e444e80e98a8f3761a22ad846145591260e3393b90c299a6b` |
| `data/upstream/strong_routing_canonical_v1.json` | 118562 | `18b337ab707dd94c792fd74721f168b637dde09e0889313db9574fa66002a3a4` |
| `data/upstream/strong_routing_budget_canonical_v1.json` | 117810 | `0a5f9c9d95e1b0717cb23a3656514479fc1cc8014c71033910836f20924069f3` |

## Figure and table mapping

| Manuscript item | Derived source |
|---|---|
| Table 1 | `canonical_v3/main_results.csv` and `canonical_v3/raw_results.csv` |
| Fig. 1a | policy means in `canonical_v3/main_results.csv` |
| Fig. 1b | paired-difference means and intervals in `canonical_v3/main_results.csv` |
| Budget-sensitivity figure | `canonical_v3_budget/budget_sensitivity.csv` |
| Supplementary primary-effects table | `canonical_v3/raw_results.csv` |
| Supplementary budget table | `canonical_v3_budget/budget_sensitivity.csv` |

## FAIR/readiness actions before public deposit

- Add the exact code commit or archived release identifier that generated these
  files.
- Add a software environment lock file and package versions.
- Add column definitions, units, allowed values, and missing-value conventions
  for each CSV.
- Add an explicit licence for the authors' derived tables and code; confirm
  that it does not purport to relicense third-party benchmark images.
- Record benchmark versions, access dates, and licence/source URLs.
- Test the repository landing page and reviewer-access route outside the author
  account before submission.

## Persistent-identifier status

The public release branch is `sivip_rebuild`; the Git commit containing this
manifest is the immutable submission-matched identifier. No DOI, archive
accession, or signed release tag has yet been assigned.
