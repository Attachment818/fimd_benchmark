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

Historical JPEGs already contain their control-point rings. Therefore their
marker size cannot be normalized losslessly to the smaller rings in newly
reproduced PNGs. No marker removal, inpainting, or manual point adjustment is
performed.
