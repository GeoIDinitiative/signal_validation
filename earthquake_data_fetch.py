import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime

# ==========================================================
# USER PARAMETERS
# ==========================================================

STARTTIME = "2022-11-11"
ENDTIME   = "2023-08-30"
MINMAG    = 3

# Mount Etna reference (can adjust slightly if needed)
ETNA_LAT = 37.751
ETNA_LON = 15.004

OUTFILE = "earthquakes_M2plus_global_etna_20221111_20230830.csv"

# ==========================================================
# FUNCTIONS
# ==========================================================

def download_fdsn_csv(url, params, source_name):
    """Download FDSN CSV and tag with source"""
    print(f"Downloading from {source_name}...")
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df["source"] = source_name
    return df


def haversine(lat1, lon1, lat2, lon2):
    """Distance (km) between two lat/lon points"""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# ==========================================================
# DOWNLOAD GLOBAL (USGS)
# ==========================================================

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

usgs_params = {
    "format": "csv",
    "starttime": STARTTIME,
    "endtime": ENDTIME,
    "minmagnitude": MINMAG,
    "orderby": "time-asc",
    "limit": 200000
}

usgs = download_fdsn_csv(USGS_URL, usgs_params, "USGS")

# ==========================================================
# DOWNLOAD LOCAL ETNA (INGV – CT)
# ==========================================================

INGV_URL = "https://sismoweb.ct.ingv.it/fdsnws/event/1/query"

ingv_params = {
    "format": "csv",
    "starttime": STARTTIME,
    "endtime": ENDTIME,
    "minmagnitude": MINMAG
}

ingv = download_fdsn_csv(INGV_URL, ingv_params, "INGV_Etna")

# ==========================================================
# STANDARDISE COLUMNS
# ==========================================================

def standardise(df):
    rename_map = {
        "time": "datetime",
        "origintime": "datetime",
        "mag": "magnitude",
        "ml": "magnitude",
        "lat": "latitude",
        "lon": "longitude"
    }
    df = df.rename(columns=rename_map)
    return df


usgs = standardise(usgs)
ingv = standardise(ingv)

common_cols = [
    "datetime", "latitude", "longitude",
    "depth", "magnitude", "place", "source"
]

usgs = usgs[[c for c in common_cols if c in usgs.columns]]
ingv = ingv[[c for c in common_cols if c in ingv.columns]]

# ==========================================================
# MERGE + DEDUPLICATE
# ==========================================================

combined = pd.concat([usgs, ingv], ignore_index=True)

# Convert time to datetime
combined["datetime"] = pd.to_datetime(combined["datetime"], errors="coerce")

# Remove exact duplicates
combined = combined.drop_duplicates(
    subset=["datetime", "latitude", "longitude", "magnitude"]
)

# ==========================================================
# DISTANCE FROM ETNA
# ==========================================================

combined["distance_km"] = haversine(
    ETNA_LAT, ETNA_LON,
    combined["latitude"].values,
    combined["longitude"].values
)

# Sort chronologically
combined = combined.sort_values("datetime")

# ==========================================================
# EXPORT
# ==========================================================

combined.to_csv(OUTFILE, index=False)

print("\n====================================")
print("DONE")
print(f"Total events: {len(combined)}")
print(f"Saved to: {OUTFILE}")
print("====================================")
