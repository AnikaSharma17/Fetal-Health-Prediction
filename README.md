# Fetal Health Prediction

Classifies cardiotocogram (CTG) records as **Normal**, **Suspect** or **Pathological** from 21 features
extracted from fetal heart rate and uterine contraction signals. Every number below comes from a held-out
test set of real records from recordings the model never saw, with confidence intervals. The evaluation
is designed to avoid the leakage that commonly inflates results on this dataset.

> Research and portfolio project. It is not a medical device and has not been clinically validated.

## Results

The selected model is **gradient boosting with class weights**. It is reported at two operating points:

- **Most likely class**: the model's own prediction.
- **Pathological threshold**: calibrated probabilities, and Pathological is flagged whenever its
  probability passes a threshold. The threshold is chosen on the training set to reach 95% Pathological
  recall.

### Across 10 train/test splits

The whole procedure (model search, threshold, test evaluation) was repeated on 10 recording-grouped
splits by `scripts/repeated_splits.py`. Gradient boosting was selected on all 10, with class weights on 9
and SMOTE on 1. Mean, with the lowest and highest split in brackets:

| Metric | Most likely class | Pathological threshold |
|---|---|---|
| Macro-F1 | **0.831** (0.779–0.889) | 0.788 (0.743–0.857) |
| Balanced accuracy | 0.831 (0.793–0.891) | 0.826 (0.795–0.873) |
| Accuracy | 0.902 (0.868–0.943) | 0.880 (0.839–0.922) |
| Pathological recall | 0.783 (0.686–0.914) | **0.937** (0.857–1.000) |
| Pathological precision | 0.790 (0.686–0.963) | 0.577 (0.422–0.680) |
| Suspect recall | 0.773 (0.661–0.864) | 0.621 (0.475–0.690) |
| Suspect precision | 0.784 (0.610–0.922) | 0.883 (0.725–1.000) |

**The notebook's split (seed 42) is the most favourable of the ten:** it has the highest macro-F1 under
both rules. A fair expectation on new recordings is about 0.83 macro-F1, not 0.89. The threshold reached
its 95% recall target on 5 of the 10 test sets and stayed at or above 0.857 on all of them. The test sets of
different splits overlap, so this spread is a rough guide rather than 10 independent trials.

### On the notebook's split

Test set: 423 records (330 Normal, 58 Suspect, 35 Pathological) from 72 recordings, 9 of which contain
Pathological segments. The Pathological threshold here is 0.195.

| Metric (95% CI) | Most likely class | Pathological threshold | Majority-class baseline |
|---|---|---|---|
| Macro-F1 | **0.889** (0.792–0.950) | 0.857 (0.713–0.946) | 0.292 |
| Balanced accuracy | 0.847 (0.731–0.941) | 0.873 (0.808–0.936) | 0.333 |
| Accuracy | 0.943 (0.909–0.976) | 0.922 (0.861–0.971) | 0.780 |
| Pathological recall | 0.743 (0.421–1.000) | **0.971** (0.948–1.000) | 0 |
| Pathological precision | 0.963 (0.800–1.000) | 0.680 (0.263–1.000) | 0 |
| Suspect recall | 0.810 (0.644–0.948) | 0.690 (0.493–0.851) | 0 |
| Suspect precision | 0.922 (0.826–1.000) | 1.000 (1.000–1.000) | 0 |

Intervals come from a bootstrap that resamples whole test recordings, since segments of one recording
are correlated.

What the numbers say:
- **The threshold trades false alarms for missed cases.** The threshold rule catches 34 of 35 Pathological
  segments instead of 26. The cost is 16 false Pathological flags, 14 of them on Normal segments, so about
  a third of its Pathological flags are wrong. Separately, calibration moves 16 Suspect segments to Normal,
  which is why Suspect recall drops.
- **Pathological numbers are uncertain.** The 35 Pathological test segments come from only 9 recordings,
  hence intervals like 0.42–1.00 for recall at the default operating point.
- **Accuracy is misleading here.** Predicting "Normal" for everything already gets 0.78, so models are
  selected and reported on macro-F1, which weighs the three classes equally.
- **Suspect is much better than before, but varies by split.** Suspect precision was 0.57 with the
  previous model (Random Forest + SMOTE, evaluated on a slightly different split). It is 0.92 on this
  split and 0.78 on average across the 10 splits (0.61–0.92). The comparison below shows SMOTE was not
  the best choice for most models.

### Model and imbalance strategy comparison

Grouped 5-fold CV on the training set, best macro-F1 per model under each way of handling the class
imbalance (SMOTE oversampling, `class_weight="balanced"`, or nothing):

| Model | SMOTE | Class weights | None |
|---|---|---|---|
| Gradient Boosting | 0.846 | **0.854** | 0.850 |
| Random Forest | 0.809 | 0.797 | 0.801 |
| Decision Tree | 0.704 | 0.693 | 0.805 |
| AdaBoost | 0.786 | n/a | 0.799 |
| SVM (RBF) | 0.799 | 0.793 | 0.795 |
| Logistic Regression | 0.767 | 0.772 | 0.778 |
| KNN | 0.754 | n/a | 0.752 |
| Majority-class baseline | n/a | n/a | 0.292 |

Gradient boosting wins under every strategy. SMOTE is the best strategy for Random Forest, SVM and KNN,
but by at most 0.008, well within the CV spread (standard deviations of 0.04–0.08). It hurts the decision
tree badly (0.70 vs 0.81).

### Calibration

SMOTE and class weights both distort predicted probabilities, so a sigmoid calibrator was fit on held-out
recordings within the training set. It lowered the out-of-fold log loss from 0.279 to 0.254 (isotonic:
0.321). **On the test set it made the probabilities worse, though** (log loss 0.147 → 0.200, Brier
0.085 → 0.107). The class-weighted gradient boosting model was already reasonably calibrated, and this
evidence does not support adding a calibration step. The threshold works on the calibrated probabilities
because it was chosen on them.

### What drives the predictions

SHAP values for the Pathological output (`fetal_health_prediction.ipynb`, "What drives the predictions?")
rank these features highest:
1. the heart-rate histogram median and mean (lower values push towards Pathological);
2. abnormal short-term variability (more pushes towards Pathological);
3. baseline heart rate (higher pushes towards Pathological).

Prolonged decelerations matter rarely but strongly when present. These are associations the model
learned, not causal effects.

## How the evaluation avoids leakage

An earlier version of this notebook reported 96.7% accuracy (and this README claimed ~98%). That number
was inflated: SMOTE oversampling was applied to the whole dataset *before* the train/test split, so more
than half of the test rows were synthetic points interpolated from training data. The current pipeline
fixes this and two related problems:

1. **Split first, handle imbalance inside training only.** Scaling, SMOTE (when used) and the classifier
   form one `imblearn` pipeline, so they are fit only on each training fold. The test set is never
   resampled. Calibration and the threshold are also chosen with cross-validation on the training set.
2. **Keep recordings together.** The 2,126 rows are segments of 352 CTG recordings, and 56% of segment
   pairs from the same recording overlap in time. The Kaggle CSV has no recording ID, so the original UCI
   release was used to recover it (see Dataset). Every split keeps all segments of a recording on one side.
   On the training set, ordinary stratified CV scores 0.906 macro-F1 and CV grouped by recording scores
   0.854, so ignoring recordings would overstate performance by about 0.05.
3. **Drop exact duplicates** (13 rows, each a repeat of a segment from the same recording) so no
   record appears twice.

An earlier version guessed recordings from runs of rows sharing a baseline value. Checked against the real
IDs, that proxy split 65 recordings across groups (adjusted Rand index 0.77), which let some leakage through.

These rules are enforced by tests (see `tests/test_train.py`).

## Dataset

`fetal_health.csv`: 2,126 CTG segments, 21 numeric features, label `fetal_health`
(1 = Normal, 2 = Suspect, 3 = Pathological, as classified by expert obstetricians).

- Classes are imbalanced: 77.8% Normal, 13.9% Suspect, 8.3% Pathological.
- No missing values. 13 exact duplicate rows are dropped, leaving 2,113.
- 2 pairs of records have identical features but different labels. They are kept and reported.
- 88 of the 352 recordings contain segments with different labels.
- Features cover baseline heart rate, accelerations, fetal movements, uterine contractions, decelerations,
  short- and long-term variability, and summary statistics of the heart-rate histogram.

`recording_ids.csv`: for each row of `fetal_health.csv`, in the same order, it gives:
- the source recording;
- its date;
- the segment file;
- the segment's start and end instants.

It is taken from the original UCI release (`CTG.xls`, "Raw Data" sheet) by
`scripts/build_recording_ids.py`. That script first checks that every feature and label matches the
Kaggle CSV row by row.

Sources:
- [Fetal Health Classification](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) on Kaggle.
- Campos, D. & Bernardes, J. (2000). Cardiotocography [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C51S4N (CC BY 4.0).
- Original study: Ayres-de-Campos D., Bernardes J., Garrido A., Marques-de-Sá J., Pereira-Leite L.,
  "SisPorto 2.0: A Program for Automated Analysis of Cardiotocograms", *Journal of Maternal-Fetal Medicine*
  9(5):311–318, 2000.

## Method

1. Load and validate the CSV (schema, missing values, plausible value ranges, valid labels) and attach
   recording IDs.
2. Exploratory analysis: class balance, correlations, distributions.
3. Drop exact duplicates.
4. Recording-grouped, stratified 80/20 train/test split.
5. Grid-search seven models plus a majority-class baseline. Each model is tried with SMOTE, class
   weights and no correction (where supported), using grouped stratified 5-fold CV scored on macro-F1.
6. Refit the best model on the full training set and evaluate it once on the test set:
   - per-class metrics with recording-level bootstrap intervals;
   - a confusion matrix;
   - a comparison with the baseline.
7. Calibrate probabilities, choose a Pathological threshold on out-of-fold training predictions, and
   evaluate that decision rule on the test set.
8. Explain predictions with SHAP, and compare stratified, recording-grouped and proxy-grouped CV.
9. Repeat steps 4–7 on 10 different splits (`scripts/repeated_splits.py`) to see how much the results
   depend on the split.

## Project structure

```
fetal_health_prediction.ipynb   # the full analysis, top to bottom
fetal_health.csv                # dataset (Kaggle)
recording_ids.csv               # source recording of each row (UCI release)
results/
    repeated_splits.csv         # one row per train/test split from scripts/repeated_splits.py
scripts/
    build_recording_ids.py      # rebuilds recording_ids.csv from the UCI CTG.xls
    repeated_splits.py          # reruns selection and evaluation on 10 splits
src/fetal_health/
    data.py                     # loading, validation, deduplication, recording groups
    train.py                    # leak-free pipelines, imbalance strategies, holdout split, model search
    evaluate.py                 # per-class metrics, bootstrap intervals, CV protocol comparison
    decision.py                 # probability calibration and the Pathological threshold
    explain.py                  # SHAP explanations
    experiment.py               # the full procedure on one split, and the summary across splits
tests/                          # pytest suite (data, leakage, reproducibility, decisions, baseline gate)
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
if you don't already have it. A full run of the notebook took 2–6 minutes on a 32-core desktop; most of
it is the model search.

The notebook only reads `results/repeated_splits.csv`. To recompute it, run
`python scripts/repeated_splits.py` (about 30 minutes for the 10 splits). It saves each split as it
finishes and skips splits already in the file, so delete the file first for a fresh run.

`recording_ids.csv` is already in the repository. To rebuild it, download `CTG.xls` from the
[UCI repository](https://archive.ics.uci.edu/dataset/193/cardiotocography), then run
`pip install xlrd` and `python scripts/build_recording_ids.py path/to/CTG.xls`.

## Limitations

- **One dataset.** There is no external validation. Each test set has only 8–10 recordings with
  Pathological segments, and macro-F1 moves by a standard deviation of 0.04 between splits. The bootstrap
  intervals and the repeated splits make this uncertainty visible, but they cannot shrink it.
- **Labels are expert judgements.** "Suspect" in particular is a borderline category on which clinicians
  often disagree, and 88 recordings have segments with different labels.
- **Calibration did not help on the test set.** The calibrated probabilities were worse there than the
  raw ones, so neither should be read as an exact risk.
- **The threshold is set for one goal, and only partly reaches it.** It targets 95% Pathological recall
  on training data, which the test set matched on 5 of 10 splits. Any other balance between missed cases
  and false alarms needs a different threshold.
- **The repeated splits fix the calibration method.** They use sigmoid calibration on every split
  instead of choosing it per split as the notebook does (the notebook's split chose sigmoid too).
- **SHAP shows associations, not causes.** The SHAP values explain the uncalibrated model in log-odds,
  not the calibrated probabilities.

## Next steps

- Validate on an independent CTG dataset with the same features.
- Find out why Suspect precision varies so much between splits (0.61–0.92).
- Choose the threshold from explicit costs of missed Pathological cases and false alarms, possibly with
  a second threshold for Suspect.
