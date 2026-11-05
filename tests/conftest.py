from pathlib import Path

import pytest

from fetal_health.data import drop_duplicates, load_data, recording_groups, split_features

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "fetal_health.csv"
RECORDINGS_PATH = ROOT / "recording_ids.csv"


@pytest.fixture(scope="session")
def raw_df():
    return load_data(DATA_PATH, RECORDINGS_PATH)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return drop_duplicates(raw_df)


@pytest.fixture(scope="session")
def xyg(clean_df):
    X, y = split_features(clean_df)
    return X, y, recording_groups(clean_df)
