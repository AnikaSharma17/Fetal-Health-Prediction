"""Repeat model selection and test evaluation over several recording-grouped splits.

Writes one row per split to results/repeated_splits.csv and one row per test segment to
results/repeated_predictions.csv, adding to both files as each split finishes, so an
interrupted run can be resumed: splits already in results/repeated_splits.csv are
skipped. Each split takes a few minutes (mostly the model search).

Usage:

    python scripts/repeated_splits.py              # the notebook's seed 42 plus seeds 0-8
    python scripts/repeated_splits.py 42 7 13      # specific seeds
"""
import sys
import time
from pathlib import Path

import pandas as pd

from fetal_health.data import drop_duplicates, load_data, recording_groups, split_features
from fetal_health.experiment import run_holdout, summarize_repeats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "repeated_splits.csv"
PREDICTIONS = ROOT / "results" / "repeated_predictions.csv"
DEFAULT_SEEDS = [42, *range(9)]


def completed_seeds() -> set:
    """Seeds already in OUT. Predictions of a split interrupted before its OUT row was
    written are dropped, so rerunning that split does not duplicate them."""
    done = set(pd.read_csv(OUT)["seed"]) if OUT.exists() else set()
    if PREDICTIONS.exists():
        predictions = pd.read_csv(PREDICTIONS)
        keep = predictions["seed"].isin(done)
        if not keep.all():
            predictions[keep].to_csv(PREDICTIONS, index=False)
    return done


def main(seeds):
    df = drop_duplicates(load_data(ROOT / "fetal_health.csv", ROOT / "recording_ids.csv"))
    X, y = split_features(df)
    groups = recording_groups(df)

    OUT.parent.mkdir(exist_ok=True)
    done = completed_seeds()
    for seed in seeds:
        if seed in done:
            print(f"seed {seed}: already in {OUT.name}, skipped")
            continue
        start = time.time()
        row, predictions = run_holdout(X, y, groups, seed)
        # predictions first: a seed only counts as done once its OUT row exists
        predictions.to_csv(PREDICTIONS, mode="a", header=not PREDICTIONS.exists(), index=False)
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"seed {seed}: {row['model']} ({row['strategy']}, calibration {row['calibration']}), "
              f"macro-F1 {row['default_macro_f1']:.3f}, "
              f"Pathological recall: threshold {row['threshold_recall_Pathological']:.3f}, "
              f"cost {row['cost_recall_Pathological']:.3f} "
              f"[{time.time() - start:.0f}s]", flush=True)

    runs = pd.read_csv(OUT)
    runs = runs[runs["seed"].isin(seeds)]
    print(f"\n{len(runs)} splits")
    print(summarize_repeats(runs).round(3).to_string())


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or DEFAULT_SEEDS)
