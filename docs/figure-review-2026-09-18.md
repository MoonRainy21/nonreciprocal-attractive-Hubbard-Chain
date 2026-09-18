# Publication figure review, 2026-09-18

Reviewed all four main-text figures, S1-S4, and both S3 split exports.
The design follows the Physical Review guidance on legibility at final column
width, proportional lettering and symbols, and readable line weights:
https://cdn.journals.aps.org/files/styleguide-pr.pdf

## Changes

| Figure | Revision |
| --- | --- |
| 1 (manuscript TikZ) | Neutral rectangular boxes, consistent sans-serif labels, and arrow labels clear of box edges. |
| 2 | Shared typography and lighter grids; preserved overlaid covariance comparisons and their marker encoding. |
| 3 | External lower-panel legends; identical mean/maximum color and hatch encoding for both coordinates; removed full-interval distance from the cropped panel. |
| 4 | Focused trajectory window; distinct line styles; legends outside curves; a dedicated header above the two control plots; padding below zero so control markers remain visible. |
| S1 | Linear axes with explicit scientific multipliers for narrow-range errors; scan ticks at sampled q values; external error legends. |
| S2 | Enlarged spectral window and useful magnitude ranges, retaining the full data in the processed table; compact shared direction legend. |
| S3 and split exports | Shorter d_min/t label, three readable eigenvalue-separation ticks, threshold annotations away from both curves and reference lines. |
| S4 | External filling, branch, projector, and grouped-bar legends; moved the plotting-floor explanation to the existing caption. |

All numerical exports remain available as vector PDF and 600-dpi PNG.
The local manuscript now includes vector PDFs instead of PNGs. Its captions
identify cropped windows and distinguish them from the full intervals used for
collapse statistics. The Fig. 4 profile text now agrees with the stored snapshot
coordinates, chi = 0.05318 and 0.14266 (previous text: 0.032 and 0.087).
No numerical data or solver results were changed.

## Color refinement

All nine numerical exports use a shared subset of Paul Tol's bright palette:
blue `#4477AA`, red `#EE6677`, green `#228833`, and purple `#AA3377`.
These are a design choice, not prescribed PRE colors. APS guidance requires
figures to remain intelligible in both color and grayscale:
https://journals.aps.org/authors/guide-acceptable-color-online-figures-h24
Neutral gray and hatching distinguish reference bars. The extra L=20 branch
uses charcoal in both Fig. 3 and S4; PBC uses purple in both Fig. 4 and S4.
S2 directional curves now also carry q-specific markers so direction dashes
and q labels remain separable without color. Figure 1 remains neutral.
All nine regenerated PDFs were rendered and visually reviewed; Fig. 3 and
Fig. 4 were additionally rendered in grayscale. Main and Supplement rebuilt
successfully. Data, axis limits, and numerical results were unchanged.

## Verification

- Regenerated all nine numerical figure exports.
- Viewed every numerical figure, including the two S3 split versions.
- Built the main manuscript and Supplement successfully.
- Rendered and visually inspected main-text figure pages 4, 7, 8, 9, and the
  duplicate appendix figure pages 14 and 16; Supplement figure pages 3, 4, 5, 7.
- Ran all six existing assertions in `tests/test_publication_figure_data.py`
  directly through `runpy`; all passed. This was not a full solver-suite run.
- `git diff --check` passed.

The manuscript files are in the adjacent `paper_source` workspace, outside this
code repository. Existing REVTeX float-placement and bibliography warnings
remain; the rendered figure pages have no observed label/data overlaps or
clipped zero markers. The main manuscript also retains its existing duplicate
appendix copies of S1-S3 as Figs. 5-7; submission packaging is a separate decision.
