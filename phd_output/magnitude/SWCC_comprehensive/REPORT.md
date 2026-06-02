# Etna tilt template search — consolidated report

## Pipeline (Design B, `run_pipeline.py`)
denoise → SWCC plots → detection → injection → network → analysis → credibility → this report

## Coverage
- 85–98% of the post-earthquake record analysed (vs 28% in the old cut-first pipeline)

## Detections & significance
- 302 per-station 'significant' windows (≈ the 1% chance rate of the 99th-pct floor)
- cross-station synchrony: smallest p = 0.055 → **no significant coincident signal**

## Sensitivity (experiment, SNR₉₀)
- SWCC magnitude: 1.08
- Net magnitude: 0.47
- SUBSPACE: 1.35
- ENVELOPE: 2.95

## Conclusion
With ~the full signal analysed and contamination controlled, the simulation templates
show **no credible correlation** with the denoised data — confirmed across single-station,
network-stacked, and four independent method families, down to the quoted SNR limits.
The pipeline recovers injected templates (validating the search), so this is a true null.