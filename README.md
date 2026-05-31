# tilt_validation — clean-room signal-validation pipeline

A from-scratch rebuild of the Etna tilt template search, **independent of
`etna_signals_phd`**, reordered around the lessons learned from that project.

## Reads (external inputs only — nothing from etna_signals_phd)
- observed tilt: `/home/owen/Signals/experiment/...` (INGV feathers, EC1.csv, school-data)
- simulation tilt: `/home/owen/Signal_Validation/solid_dofs/tilt/sim{1..4}/tilt/<station>.txt`
- earthquake catalogue: `earthquakes_utc.csv` (copied in — self-contained)

## The core change vs the old project: order of operations
The old pipeline perfected a detector first and discovered its two biggest limitations
last (the template bank barely covers anything; the contamination model missed 57% of
in-band earthquakes). This rebuild front-loads the questions that decide whether the
search can work at all, and fixes the pipeline before looking at the data.

```
00 characterize   noise (PSD/color/glitches) + templates + BANK COVERAGE (fitting factor)
                  → GATE: if the bank can't see plausible signals, say so before going further
01 validate       closed-box: matched-filter sensitivity (injection-recovery), null floors,
                  time-slide background — all fixed on simulations BEFORE touching real data
02 condition      data-driven conditioning: bandpass + despike + empirical contamination
                  flagging (STA/LTA), catalogue used only as corroboration
03 search         run the FIXED pipeline on the real data — once
04 report         result + honest caveats (bank coverage, template realism)
```

## Principles (the things I'd do differently)
1. **Coverage first.** Compute the fitting factor before building a detector — it bounds
   what any search can find. A null with a narrow bank is a weak statement; know that early.
2. **Interrogate the templates.** The templates are the simulated tilt itself — treat that
   as an assumption to test (realism, duration, how to window), not a given.
3. **Closed-box.** Fix every threshold/parameter on simulations + injections, then run on
   data once. No tuning to the real data.
4. **Data-driven contamination.** Flag contamination from the signal (STA/LTA, glitch
   statistics); use the earthquake catalogue only to corroborate, not to decide.
5. **Honest scoping.** The bandpass is a bandpass — only call it environmental cleaning if
   real auxiliary (pressure/temperature) data is used.

## Status
`00_characterize.py` implemented (the gate). 01–04 are specified above and built on the
same independent `lib.py`. This folder never imports from `etna_signals_phd`.
