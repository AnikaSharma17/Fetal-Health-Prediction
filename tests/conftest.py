from pathlib import Path

import pytest

from fetal_health.data import drop_duplicates, load_data, pseudo_recording_groups, split_features

DATA_PATH = Path(__file__).resolve().parents[1] / "fetal_health.csv"


@pytest.fixture(scope="session")
def raw_df():
    return load_data(DATA_PATH)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    return drop_duplicates(raw_df)


@pytest.fixture(scope="session")
def xyg(clean_df):
    X, y = split_features(clean_df)
    return X, y, pseudo_recording_groups(clean_df)
