"""
plot_labels.py — central DISPLAY labels + legend policy for thesis-standard figures.

EEC1 and EC1 are the SAME borehole site recorded in two campaigns; we label them by PERIOD so the
pairing is explicit and unambiguous (internal station CODES in the data/paths are unchanged):
    EEC1  ->  "EC1 (eruptive)"        (INGV, winter 2022-23, spans the eruptive period)
    EC1   ->  "EC1 (non-eruptive)"    (experiment, summer 2023)
"""
STATION_LABEL = {
    "EEC1": "EC1 (eruptive)",
    "EC1":  "EC1 (non-eruptive)",
    "ECPN": "ECPN", "EC10": "EC10", "ECIT": "ECIT", "ECOR": "ECOR", "EMAS": "EMAS",
}
DATASET_LABEL = {"ingv": "eruptive period", "experiment": "non-eruptive period"}


def slab(s):
    return STATION_LABEL.get(s, s)


def slabs(lst):
    return [slab(s) for s in lst]


def dlab(d):
    return DATASET_LABEL.get(d, d)


def legend_or_hide(ax, max_items=7, **kw):
    """Thesis legend policy: show a legend only if it has a small, non-clashing number of entries;
    otherwise omit it (the axis/tick labels carry the meaning). Returns True if a legend was drawn."""
    handles, labels = ax.get_legend_handles_labels()
    if 0 < len(handles) <= max_items:
        ax.legend(**{"fontsize": 9, "framealpha": 0.9, "loc": "best", **kw})
        return True
    return False
