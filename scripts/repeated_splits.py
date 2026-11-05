"""Repeat model selection and test evaluation over several recording-grouped splits.

Writes one row per split to results/repeated_splits.csv, adding to the file as each
split finishes, so an interrupted run can be resumed: splits already in the file are
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
DEFAULT_SEEDS = [42, *range(9)]


def main(seeds):
    df = drop_duplicates(load_data(ROOT / "fetal_health.csv", ROOT / "recording_ids.csv"))
    X, y = split_features(df)
    groups = recording_groups(df)

    OUT.parent.mkdir(exist_ok=True)
    done = set(pd.read_csv(OUT)["seed"]) if OUT.exists() else set()
    for seed in seeds:
        if seed in done:
            print(f"seed {seed}: already in {OUT.name}, skipped")
            continue
        start = time.time()
        row = run_holdout(X, y, groups, seed)
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"seed {seed}: {row['model']} ({row['strategy']}), "
              f"macro-F1 {row['default_macro_f1']:.3f}, "
              f"Pathological recall with threshold {row['threshold_recall_Pathological']:.3f} "
              f"[{time.time() - start:.0f}s]", flush=True)

    runs = pd.read_csv(OUT)
    runs = runs[runs["seed"].isin(seeds)]
    print(f"\n{len(runs)} splits")
    print(summarize_repeats(runs).round(3).to_string())


if __name__ == "__main__":
    main([int(s) for s in sys.argv[1:]] or DEFAULT_SEEDS)
