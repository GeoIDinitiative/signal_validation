# Tilt-template validation — thesis summary

Validation of the numerically simulated tilt templates against the Etna borehole-tiltmeter
records was carried out with a single segment-aware pipeline following a "condition-the-full-record,
then veto" design: the complete record (teleseismic P-wave arrivals already excised) was thermally
de-noised, band-pass filtered, and only afterwards were earthquake- and anomaly-contaminated windows
flagged for a post-correlation veto, retaining ~100% of the record against ~28% for a cut-first
scheme. Thermal noise was removed by coherent-admittance subtraction — a coherence-gated complex
transfer function H(f)=S_{tilt,na}/S_{na,na} estimated against the instrument's internal
temperature/level (`na`) channel — and the signal was band-pass filtered to 0.001–0.01 Hz
(100–1000 s) with a zero-phase fourth-order Butterworth filter, the same band occupied by the
templates (97% of template power in-band). The correction removed 42% of the in-band RMS at station
EEC1, where the `na` channel is strongly coherent with the tilt (in-band r = −0.81), and was
correctly negligible elsewhere, the `na` channel carrying no coherent in-band signal at the other
stations. A vectorised, gap-aware sliding-window cross-correlation against the four-simulation,
sixteen-template bank, assessed against two-tier null floors from phase-randomised surrogates,
returned no significant detection in either the winter (INGV) or summer (experiment) campaign. The
null was robust to every relaxation tested: cross-station synchrony was non-significant at all
coincidence windows from 10 to 90 min (spanning the simulations' own 15–64 min across-station peak
spread); densifying the bank by time-warping and inter-simulation interpolation revealed no hidden
matches; and a network detector matched to the simulations' predicted staggered inter-station timing
failed to exceed chance for any of the sixteen patterns (p ≥ 0.22). End-to-end validity was confirmed
by injection–recovery (high-SNR injections recovered at unit probability), a time-slide false-alarm
analysis (loudest coincidence consistent with background), and a χ² signal-consistency test that
identified the few chance detections as glitch-like.

The null was therefore expressed as a quantitative upper limit: any transient sharing the templates'
spectral shape with peak amplitude below A_min = SNR₉₀ · σ_noise would have been recovered with 90%
confidence, giving per-station limits of ≈0.003–0.07 instrument units (SNR₉₀ ≈ 1.0–1.1). Finally,
exploiting the fact that stations EEC1 and EC1 are the same borehole instrument recorded in the two
campaigns (confirmed by an identical digitisation step), the winter record was found to carry ≈6× the
in-band tilt energy and ≈7.5× the overall tilt excursion of the summer record — a real elevation in
ground tilt rather than an instrumental or calibration artefact, most plausibly attributable to
seasonally stronger meteorological (barometric and wind) loading, which accordingly weakens the
detection upper limit for the winter campaign and which the available data cannot further separate
into environmental and volcanic components.
