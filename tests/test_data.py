import numpy as np
import pandas as pd
import pytest

from conftest import DATA_PATH, RECORDINGS_PATH
from fetal_health.data import (FEATURES, RECORDING, TARGET, drop_duplicates, label_conflicts,
                               load_data, pseudo_recording_groups, recording_groups, validate)


def test_real_dataset_passes_validation(raw_df):
    assert raw_df.shape == (2126, 23)  # 21 features, label, recording
    assert set(raw_df[TARGET].unique()) == {1, 2, 3}


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.drop(columns="histogram_mean"), "missing columns"),
    (lambda d: d.assign(accelerations="high"), "non-numeric"),
    (lambda d: d.assign(histogram_mode=np.nan), "missing values"),
    (lambda d: d.assign(**{"baseline value": 500.0}), "plausible range"),
    (lambda d: d.assign(histogram_tendency=0.5), "histogram_tendency"),
    (lambda d: d.assign(**{TARGET: 4.0}), "unknown labels"),
])
def test_validation_rejects_bad_input(raw_df, mutate, message):
    with pytest.raises(ValueError, match=message):
        validate(mutate(raw_df.head(20).copy()))


def test_drop_duplicates(raw_df):
    clean = drop_duplicates(raw_df)
    assert len(clean) == 2113
    assert not clean.duplicated().any()
    # first occurrences keep their original order
    pd.testing.assert_frame_equal(clean.head(68), raw_df.head(68))


def test_duplicates_ignore_the_recording_column(raw_df):
    """Duplicates are judged on features and label only."""
    twin = raw_df.head(1).assign(**{RECORDING: "another.txt"})
    assert len(drop_duplicates(pd.concat([raw_df.head(1), twin]))) == 1


def test_label_conflicts_are_reported(clean_df):
    conflicts = label_conflicts(clean_df)
    assert len(conflicts) == 4
    assert conflicts.groupby(FEATURES)[TARGET].nunique().eq(2).all()


def test_recording_ids_are_attached(raw_df):
    assert raw_df[RECORDING].notna().all()
    assert raw_df[RECORDING].nunique() == 352


def test_recording_ids_must_match_row_count(tmp_path):
    short = tmp_path / "ids.csv"
    pd.read_csv(RECORDINGS_PATH).head(10).to_csv(short, index=False)
    with pytest.raises(ValueError, match="row for row"):
        load_data(DATA_PATH, short)


def test_recording_groups_require_ids():
    with pytest.raises(ValueError, match="recordings_path"):
        recording_groups(pd.DataFrame({"baseline value": [120]}))


def test_pseudo_groups_follow_baseline_runs():
    df = pd.DataFrame({"baseline value": [120, 120, 130, 130, 130, 120]})
    assert pseudo_recording_groups(df).tolist() == [1, 1, 2, 2, 2, 3]
