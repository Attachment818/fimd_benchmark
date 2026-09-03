# FIMD02 qualitative figure

Panel order:

1. Target and Source (historical export)
2. SIFT (reproduced in this repository)
3. NCNet PF-Pascal checkpoint (reproduced; near-identity failure case)
4. SuperPoint (reproduced)
5. GeoFormer (reproduced)
6. SuperRetina (historical export)
7. RetinaRegNet / RRN (historical export)
8. Ours (historical export)

The historical images in `supplementary_experiments/fimd` are 3888x3888 even
though the original FIMD02 reference and query images have a 3:2 aspect ratio.
The composition script restores those exports to 3:2 for presentation only. It
does not use the resized historical panels to recompute any metric.

The final 4x2 composition normalizes all control-point rings to a seven-pixel
radius and two-pixel line width at the 900x600 panel resolution. Existing ring
pixels are detected and locally repaired before the common markers are drawn.
Green reference centers come directly from the FIMD02 annotation; the first
panel's red query centers also come from the annotation. For method panels, red
predicted centers are detected from the existing overlays. No center is moved
manually, and this presentation-only normalization does not alter transforms or
metrics. The original method and historical overlays remain unchanged.
