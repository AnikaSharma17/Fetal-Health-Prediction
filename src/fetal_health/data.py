"""Loading, validation and cleaning of the CTG dataset."""
from pathlib import Path

import pandas as pd

TARGET = "fetal_health"
CLASS_NAMES = {1: "Normal", 2: "Suspect", 3: "Pathological"}

# Physically plausible bounds per feature (inclusive). They are deliberately wide:
# the goal is to catch unit errors and corrupted rows, not to encode clinical norms.
BPM = (30, 260)
PER_SECOND = (0, 1)
PERCENT = (0, 100)
FEATURE_RANGES = {
    "baseline value": BPM,
    "accelerations": PER_SECOND,
    "fetal_movement": PER_SECOND,
    "uterine_contractions": PER_SECOND,
    "light_decelerations": PER_SECOND,
    "severe_decelerations": PER_SECOND,
    "prolongued_decelerations": PER_SECOND,
    "abnormal_short_term_variability": PERCENT,
    "mean_value_of_short_term_variability": (0, 100),
    "percentage_of_time_with_abnormal_long_term_variability": PERCENT,
    "mean_value_of_long_term_variability": (0, 100),
    "histogram_width": (0, 260),
    "histogram_min": BPM,
    "histogram_max": BPM,
    "histogram_number_of_peaks": (0, 100),
    "histogram_number_of_zeroes": (0, 100),
    "histogram_mode": BPM,
    "histogram_mean": BPM,
    "histogram_median": BPM,
    "histogram_variance": (0, 1000),
    "histogram_tendency": (-1, 1),
}
FEATURES = list(FEATURE_RANGES)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError if df does not match the expected schema and value ranges."""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    cols = FEATURES + [TARGET]
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"non-numeric columns: {non_numeric}")

    nulls = df[cols].isna().sum()
    if nulls.any():
        raise ValueError(f"missing values: {nulls[nulls > 0].to_dict()}")

    out_of_range = {}
    for col, (low, high) in FEATURE_RANGES.items():
        bad = ~df[col].between(low, high)
        if bad.any():
            out_of_range[col] = int(bad.sum())
    if out_of_range:
        raise ValueError(f"values outside plausible range (rows per column): {out_of_range}")

    if not df["histogram_tendency"].isin([-1, 0, 1]).all():
        raise ValueError("histogram_tendency must be -1, 0 or 1")

    bad_labels = set(df[TARGET].unique()) - set(CLASS_NAMES)
    if bad_labels:
        raise ValueError(f"unknown labels: {sorted(bad_labels)}")
    return df


def load_data(path: str | Path) -> pd.DataFrame:
    """Read the CSV and validate it."""
    return validate(pd.read_csv(path))


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows, keeping the first occurrence and the original row order."""
    return df.drop_duplicates().reset_index(drop=True)


def label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose features are identical to another row but whose label differs."""
    dup = df[df.duplicated(subset=FEATURES, keep=False)]
    n_labels = dup.groupby(FEATURES)[TARGET].transform("nunique")
    return dup[n_labels > 1]


def pseudo_recording_groups(df: pd.DataFrame) -> pd.Series:
    """Approximate recording IDs as runs of consecutive rows with the same baseline value.

    The CSV has no recording identifier, but adjacent rows look like segments of the
    same CTG (they share the baseline far more often than random pairs). Keeping each
    run on one side of a split guards against that leakage. This is a heuristic proxy,
    not ground truth, and relies on the file's original row order.
    """
    baseline = df["baseline value"]
    return (baseline != baseline.shift()).cumsum().rename("group")


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with integer labels."""
    return df[FEATURES], df[TARGET].astype(int)
