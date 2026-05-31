"""
P-wave exclusion buffer — asymmetric pre/post ETA windows.

Physical basis
--------------
PRE  (before p_wave_eta):
    The p_wave_eta column already encodes origin_time + P travel time, so the
    pre-buffer only needs to cover travel-time MODEL UNCERTAINTY (typically
    ±10–20 % for teleseismic paths).  It is magnitude-dependent but NOT
    distance-dependent because that is already baked into the ETA.

POST (after p_wave_eta):
    Must cover the full seismic wave train from P-wave onset through the end
    of observable surface-wave coda in the 0.001–0.01 Hz bandpass.  Two
    physics-based contributions are computed and summed:

    1. SURFACE WAVE TRANSIT TIME  (distance-dependent)
       The slowest dominant Rayleigh wave in our band (~3.5 km/s group
       velocity) arrives at the sensor later than P by:

           t_surf = dist × (1/V_R − 1/V_P)
                  ≈ dist × (1/3.5 − 1/8.0)  s/km

       This term INCREASES with distance (the opposite of naïvely assuming
       "far = weaker = shorter").  Surface waves at 12 686 km arrive ~34 min
       after the P-wave ETA; at 1 943 km they arrive in only ~5 min.

       GATING: the transit term is only applied if the event is large enough
       to produce surface waves above the tiltmeter noise floor at that
       distance.  Empirical threshold from our diagnostic plots:

           M_min(d) = SURF_MAG_BASE + SURF_MAG_SLOPE × log₁₀(d / 1000)

       Calibrated against observations:
         • M6.8 at 17 403 km  → no visible signal → below threshold (correct)
         • M6.3 at 10 921 km  → no visible signal → below threshold (correct)
         • M6.9 at  5 021 km  → visible signal    → above threshold (correct)
         • M7.7 at 12 686 km  → clearly visible   → above threshold (correct)

    2. MAGNITUDE-DEPENDENT CODA ALLOWANCE
       After the surface wave peak, coda and post-seismic tilt persist for a
       duration that scales log-linearly with magnitude (each additional
       magnitude unit roughly triples the coda duration in this band):

           coda = CODA_BASE × 10^(CODA_LOG_SCALE × max(0, M − CODA_REF_MAG))

    Final post = pre + t_surf_gated + coda
    Capped at POST_MAX_MIN to avoid excluding hours of clean data.

Tunable constants — all at the top of this file.
"""

import numpy as np

# ── pre-buffer (before ETA) ───────────────────────────────────────────────────
PRE_M50_PLUS  = 15.0   # minutes, M ≥ 5.0
PRE_M40_M50   = 10.0   # minutes, M 4.0–5.0
PRE_SMALL     =  7.0   # minutes, M < 4.0
PRE_DEFAULT   = 10.0   # minutes, unknown magnitude

# ── surface wave physics ──────────────────────────────────────────────────────
V_P_KMS       =  8.0   # effective teleseismic P-wave velocity (km/s)
V_R_KMS       =  3.5   # dominant Rayleigh group velocity in 0.001–0.01 Hz (km/s)

# Surface wave gating: only add transit time if M > M_min(dist)
# M_min(d) = SURF_MAG_BASE + SURF_MAG_SLOPE × log10(d / 1000 km)
SURF_MAG_BASE  = 5.5   # threshold at 1 000 km
SURF_MAG_SLOPE = 1.1   # threshold steepness per decade of distance

# ── magnitude-dependent coda allowance ───────────────────────────────────────
CODA_REF_MAG    = 5.5  # magnitude below which no extra coda is added
CODA_BASE       = 2.0  # base coda duration (minutes) at M = CODA_REF_MAG
CODA_LOG_SCALE  = 0.4  # each +1 magnitude unit multiplies coda by 10^0.4 ≈ 2.5×

# ── hard cap ──────────────────────────────────────────────────────────────────
POST_MAX_MIN    = 60.0  # never exclude more than this many minutes after ETA


def calculate_pwave_buffers(distance_km, magnitude=None):
    """
    Return (pre_min, post_min) asymmetric P-wave exclusion buffer in minutes.

    Parameters
    ----------
    distance_km : float | None   Epicentral distance (km)
    magnitude   : float | None   Earthquake magnitude

    Returns
    -------
    (pre_min, post_min) : tuple of float
        pre_min  — minutes to exclude BEFORE p_wave_eta
        post_min — minutes to exclude AFTER  p_wave_eta  (≥ pre_min)
    """
    def _safe(v):
        if v is None:
            return None
        try:
            f = float(v)
            return None if np.isnan(f) else f
        except (TypeError, ValueError):
            return None

    mag  = _safe(magnitude)
    dist = _safe(distance_km)

    # ── pre-buffer ────────────────────────────────────────────────────────────
    if mag is None:
        pre = PRE_DEFAULT
    elif mag >= 5.0:
        pre = PRE_M50_PLUS
    elif mag >= 4.0:
        pre = PRE_M40_M50
    else:
        pre = PRE_SMALL

    # ── surface wave transit time (gated on minimum detectable magnitude) ─────
    t_surf_min = 0.0
    if dist is not None and dist > 0 and mag is not None:
        mag_threshold = SURF_MAG_BASE + SURF_MAG_SLOPE * np.log10(dist / 1000.0)
        if mag >= mag_threshold:
            t_surf_s   = dist * (1.0 / V_R_KMS - 1.0 / V_P_KMS)
            t_surf_min = t_surf_s / 60.0

    # ── magnitude-dependent coda allowance ────────────────────────────────────
    if mag is not None and mag > CODA_REF_MAG:
        coda_min = CODA_BASE * (10.0 ** (CODA_LOG_SCALE * (mag - CODA_REF_MAG)))
    else:
        coda_min = 0.0

    # ── post-buffer ───────────────────────────────────────────────────────────
    post = pre + t_surf_min + coda_min
    post = min(post, POST_MAX_MIN)
    post = max(post, pre)

    return pre, round(post, 1)


def symmetric_buffer(distance_km, magnitude=None):
    """Legacy shim — returns single float equal to post_min."""
    _, post = calculate_pwave_buffers(distance_km, magnitude)
    return post
