"""Build recording_ids.csv from the original UCI Cardiotocography release.

The Kaggle CSV (fetal_health.csv) has no recording identifier. The UCI release
(CTG.xls, "Raw Data" sheet) has one row per CTG segment with the source recording
(FileName), its date, the segment file and the segment's start/end instants. Its rows
are in the same order as the Kaggle CSV; this script checks that row by row before
writing anything.

Source: Campos, D. & Bernardes, J. (2000). Cardiotocography [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C51S4N (CC BY 4.0).

Usage (needs `pip install xlrd`, which is deliberately not in requirements.txt):

    python scripts/build_recording_ids.py path/to/CTG.xls
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Kaggle column -> UCI "Raw Data" column, for columns that are copied unchanged
SAME = {
    "baseline value": "LB",
    "abnormal_short_term_variability": "ASTV",
    "mean_value_of_short_term_variability": "MSTV",
    "percentage_of_time_with_abnormal_long_term_variability": "ALTV",
    "mean_value_of_long_term_variability": "MLTV",
    "histogram_width": "Width",
    "histogram_min": "Min",
    "histogram_max": "Max",
    "histogram_number_of_peaks": "Nmax",
    "histogram_number_of_zeroes": "Nzeros",
    "histogram_mode": "Mode",
    "histogram_mean": "Mean",
    "histogram_median": "Median",
    "histogram_variance": "Variance",
    "histogram_tendency": "Tendency",
    "fetal_health": "NSP",
}
# Kaggle stores event counts per second of segment, rounded to 3 decimals
PER_SECOND = {
    "accelerations": "AC",
    "fetal_movement": "FM",
    "uterine_contractions": "UC",
    "light_decelerations": "DL",
    "severe_decelerations": "DS",
    "prolongued_decelerations": "DP",
}


def main(xls_path: str) -> None:
    raw = pd.read_excel(xls_path, sheet_name="Raw Data", engine="xlrd")
    raw = raw.dropna(subset=["FileName"]).reset_index(drop=True)
    kaggle = pd.read_csv(ROOT / "fetal_health.csv")
    if len(raw) != len(kaggle):
        sys.exit(f"row count mismatch: UCI {len(raw)} vs Kaggle {len(kaggle)}")

    duration = raw["e"] - raw["b"]
    mismatched = [k for k, u in SAME.items()
                  if not np.allclose(kaggle[k], raw[u].astype(float))]
    mismatched += [k for k, u in PER_SECOND.items()
                   if not np.allclose(kaggle[k], (raw[u] / duration).round(3), atol=1e-3)]
    if mismatched:
        sys.exit(f"rows are not aligned; columns that differ: {mismatched}")

    ids = pd.DataFrame({
        "recording": raw["FileName"].str.strip(),
        "date": pd.to_datetime(raw["Date"]).dt.strftime("%Y-%m-%d"),
        "segment": raw["SegFile"].str.strip(),
        "start": raw["b"].astype(int),
        "end": raw["e"].astype(int),
    })
    out = ROOT / "recording_ids.csv"
    ids.to_csv(out, index=False)
    print(f"wrote {out.name}: {len(ids)} segments from {ids['recording'].nunique()} recordings")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
