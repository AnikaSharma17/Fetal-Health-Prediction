# Fetal Health Prediction

Classifies cardiotocogram (CTG) records as **Normal**, **Suspect** or **Pathological** from 21 features
extracted from fetal heart rate and uterine contraction signals. Every number below comes from a held-out
test set of real records, and the evaluation is designed to avoid the leakage that commonly inflates
results on this dataset.

> Research and portfolio project. It is not a medical device and has not been clinically validated.

## Results

Test set: 422 real records (329 Normal, 58 Suspect, 35 Pathological) from recordings the model never saw
during training or tuning.

| Metric | Random Forest (selected) | Majority-class baseline |
|---|---|---|
| Macro-F1 | **0.837** | 0.292 |
| Balanced accuracy | 0.886 | 0.333 |
| Accuracy | 0.882 | 0.780 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.973 | 0.881 | 0.925 |
| Suspect | 0.568 | 0.862 | 0.685 |
| Pathological | 0.889 | 0.914 | 0.901 |

What the numbers say:
- **Pathological cases are caught most of the time (recall 0.91)**, which is the class that matters most clinically.
- **Suspect is the weak spot.** Recall is good, but only 57% of "Suspect" predictions are correct; many
  Normal records get flagged. SMOTE pushes the model toward this trade-off.
- **Accuracy is misleading here.** Predicting "Normal" for everything already gets 0.78, so models are
  selected and reported on macro-F1, which weighs the three classes equally.

Model comparison (grouped 5-fold CV on the training set, macro-F1, mean ± std):

| Model | Macro-F1 |
|---|---|
| Random Forest | 0.831 ± 0.046 |
| AdaBoost | 0.819 ± 0.026 |
| SVM (RBF) | 0.804 ± 0.029 |
| Logistic Regression | 0.782 ± 0.012 |
| Decision Tree | 0.764 ± 0.062 |
| KNN | 0.748 ± 0.028 |
| Majority-class baseline | 0.292 ± 0.000 |

## How the evaluation avoids leakage

An earlier version of this notebook reported 96.7% accuracy (and this README claimed ~98%). That number
was inflated: SMOTE oversampling was applied to the whole dataset *before* the train/test split, so more
than half of the test rows were synthetic points interpolated from training data. The current pipeline
fixes this and two related problems:

1. **Split first, resample inside training only.** Scaling, SMOTE and the classifier form one
   `imblearn` pipeline, so they are fit only on each training fold. The test set is never resampled.
2. **Keep recordings together.** Adjacent rows in the CSV look like consecutive segments of the same CTG
   (they share the baseline heart rate 82% of the time, against 3% for random pairs), but the file has no
   recording ID. Runs of rows with the same baseline value are used as a proxy recording ID, and every
   split keeps a run on one side. On the training set, ordinary stratified CV scores 0.854 macro-F1 and
   grouped CV scores 0.831, so grouping removes a modest optimistic bias.
3. **Drop exact duplicates** (13 rows) so no record appears on both sides of a split.

These rules are enforced by tests (see `tests/test_train.py`).

## Dataset

`fetal_health.csv`: 2,126 CTG records, 21 numeric features, label `fetal_health`
(1 = Normal, 2 = Suspect, 3 = Pathological, as classified by expert obstetricians).

- Classes are imbalanced: 77.8% Normal, 13.9% Suspect, 8.3% Pathological.
- No missing values. 13 exact duplicate rows are dropped, leaving 2,113.
- 2 pairs of records have identical features but different labels. They are kept and reported.
- Features cover baseline heart rate, accelerations, fetal movements, uterine contractions, decelerations,
  short- and long-term variability, and summary statistics of the heart-rate histogram.

Source: the [Fetal Health Classification](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)
dataset on Kaggle, derived from the UCI Cardiotocography dataset. Original study: Ayres-de-Campos et al.,
"SisPorto 2.0: A Program for Automated Analysis of Cardiotocograms", *Journal of Maternal-Fetal Medicine*, 2000.

## Method

1. Load and validate the CSV (schema, missing values, plausible value ranges, valid labels).
2. Exploratory analysis: class balance, correlations, distributions.
3. Drop exact duplicates; build proxy recording groups.
4. Group-aware, stratified 80/20 train/test split.
5. Grid-search six models plus a majority-class baseline, each as a MinMax scaling → SMOTE → classifier
   pipeline, using grouped stratified 5-fold CV scored on macro-F1.
6. Refit the best model on the full training set and evaluate once on the test set: per-class metrics,
   confusion matrix, and comparison with the baseline.

## Project structure

```
fetal_health_prediction.ipynb   # the full analysis, top to bottom
fetal_health.csv                # dataset
src/fetal_health/
    data.py                     # loading, validation, deduplication, recording groups
    train.py                    # leak-free pipelines, holdout split, model search
    evaluate.py                 # per-class metrics, CV protocol comparison
tests/                          # pytest suite (data validation, leakage, reproducibility, baseline gate)
```

## How to run

Requires Python 3.10+ (tested with 3.12.4).

```bash
git clone https://github.com/AnikaSharma17/Fetal-Health-Prediction.git
cd Fetal-Health-Prediction
python -m venv .venv
```

Activate the environment (`source .venv/bin/activate` on macOS/Linux, `.venv\Scripts\activate` on Windows), then:

```bash
pip install -r requirements-dev.txt
pytest
jupyter notebook fetal_health_prediction.ipynb   # or open it in VS Code
```

`requirements.txt` installs the pinned runtime dependencies and the `fetal_health` package in editable
mode; `requirements-dev.txt` adds pytest. Jupyter itself is not pinned; install it with `pip install notebook`
if you don't already have it. A full run of the notebook takes about 2 minutes on a laptop.

## Limitations

- **Recording groups are a heuristic.** Without real recording IDs, some leakage between segments of the
  same CTG may remain, so the scores above may still be slightly optimistic.
- **One dataset, one test split.** There is no external validation, and a 422-row test set with 35
  Pathological cases gives wide uncertainty on per-class numbers.
- **Labels are expert judgements.** "Suspect" in particular is a borderline category on which clinicians
  often disagree.
- **No probability calibration or decision thresholds yet.** Predictions are hard labels; the model has not
  been tuned to a target Pathological recall.

## Next steps

- Recover true recording IDs from the original UCI release for exact grouped evaluation.
- Add gradient boosting and an ablation of class weights vs SMOTE vs no resampling.
- Tune decision thresholds for Pathological recall, calibrate probabilities, and explain predictions with SHAP.
