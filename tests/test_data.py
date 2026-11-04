import numpy as np
import pandas as pd
import pytest

from fetal_health.data import (FEATURES, TARGET, drop_duplicates, label_conflicts,
                               pseudo_recording_groups, validate)


def test_real_dataset_passes_validation(raw_df):
    assert raw_df.shape == (2126, 22)
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


def test_label_conflicts_are_reported(clean_df):
    conflicts = label_conflicts(clean_df)
    assert len(conflicts) == 4
    assert conflicts.groupby(FEATURES)[TARGET].nunique().eq(2).all()


def test_pseudo_groups_follow_baseline_runs():
    df = pd.DataFrame({"baseline value": [120, 120, 130, 130, 130, 120]})
    assert pseudo_recording_groups(df).tolist() == [1, 1, 2, 2, 2, 3]
