# Figure QA record

The JSON file in this directory is machine-readable delivery evidence for the
final Python/Matplotlib export of Fig. 1. It is not a manuscript upload item.

Final checks performed after the legend-layout revision:

- plotting-source preflight: 20 PASS, 0 WARN, 0 FAIL;
- rendered PDF text audit: minimum glyph size 6 pt, with a 5 pt required floor;
- rendered collision audit: PASS, 0 FAIL, 0 WARN;
- manual inspection: both panel legends are outside their plotting areas and
  remain readable in the compiled SIViP PDF at final page size;
- export formats synchronized: PDF, SVG, TIFF (600 dpi), and PNG.

The collision overlay PDF is intentionally excluded from the submission bundle
because it is a diagnostic visualization, not an author-facing figure.
