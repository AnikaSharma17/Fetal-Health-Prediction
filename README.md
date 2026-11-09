# Fetal Health Prediction

Classifies cardiotocogram (CTG) records as **Normal**, **Suspect** or **Pathological** from 21 features
extracted from fetal heart rate and uterine contraction signals. Every number below comes from a held-out
test set of real records from recordings the model never saw, with confidence intervals. The evaluation
is designed to avoid the leakage that commonly inflates results on this dataset.

> Research and portfolio project. It is not a medical device and has not been clinically validated.

## Results

The selected model is **gradient boosting with class weights**. It is reported with three decision rules:

- **Most likely class**: the model's own prediction.
- **95%-recall threshold**: Pathological is flagged whenever its probability passes a threshold chosen on
  the training set to reach 95% Pathological recall; otherwise the likelier of Normal and Suspect.
- **Cost rule**: two thresholds, one for Pathological and then one for Suspect, chosen together to minimise
  the mean misclassification cost on out-of-fold training predictions. The costs are **illustrative, not
  clinical** (see [Cost-based decisions](#cost-based-decisions)).

Both threshold rules use the model's probabilities after the calibration step (none, sigmoid or isotonic)
that has the lowest out-of-fold log loss on the training set. Nothing is chosen on the test set.

### Across 10 train/test splits

The whole procedure (model search, calibration choice, both threshold rules, test evaluation) was repeated on
10 recording-grouped splits by `scripts/repeated_splits.py`. Gradient boosting was selected on all 10, with
class weights on 9 and SMOTE on 1; sigmoid calibration was chosen on 8 splits and isotonic on 2. Mean, with
the lowest and highest split in brackets:

| Metric | Most likely class | 95%-recall threshold | Cost rule |
|---|---|---|---|
| Macro-F1 | **0.831** (0.779–0.889) | 0.789 (0.735–0.857) | 0.795 (0.736–0.861) |
| Balanced accuracy | 0.831 (0.793–0.891) | 0.829 (0.795–0.873) | **0.856** (0.822–0.892) |
| Accuracy | 0.902 (0.868–0.943) | 0.880 (0.839–0.922) | 0.875 (0.835–0.922) |
| Pathological recall | 0.783 (0.686–0.914) | **0.946** (0.857–1.000) | **0.946** (0.857–1.000) |
| Pathological precision | 0.790 (0.686–0.963) | 0.581 (0.422–0.680) | 0.592 (0.472–0.680) |
| Suspect recall | 0.773 (0.661–0.864) | 0.623 (0.475–0.746) | 0.729 (0.690–0.763) |
| Suspect precision | 0.784 (0.610–0.922) | 0.882 (0.725–1.000) | 0.755 (0.584–0.930) |
| Mean cost (illustrative costs) | 0.282 (0.151–0.378) | 0.200 (0.137–0.270) | **0.190** (0.128–0.246) |

**The notebook's split (seed 42) is the most favourable of the ten:** it has the highest macro-F1 under
all three rules. A fair expectation on new recordings is about 0.83 macro-F1, not 0.89. The 95%-recall
threshold reached its target on 6 of the 10 test sets, and so did the cost rule; both stayed at or above
0.857 Pathological recall on all of them. The cost rule had a lower mean cost than the 95%-recall threshold
on 6 splits, the same on 1 and a higher one on 3. The test sets of different splits overlap, so this
spread is a rough guide rather than 10 independent trials.

### On the notebook's split

Test set: 423 records (330 Normal, 58 Suspect, 35 Pathological) from 72 recordings, 9 of which contain
Pathological segments. The 95%-recall threshold here is 0.195; the cost rule flags Pathological at
P(Pathological) ≥ 0.19, then Suspect at P(Suspect) ≥ 0.33.

| Metric (95% CI) | Most likely class | 95%-recall threshold | Cost rule | Majority-class baseline |
|---|---|---|---|---|
| Macro-F1 | **0.889** (0.792–0.950) | 0.857 (0.713–0.946) | 0.861 (0.721–0.947) | 0.292 |
| Balanced accuracy | 0.847 (0.731–0.941) | 0.873 (0.808–0.936) | **0.892** (0.835–0.948) | 0.333 |
| Accuracy | 0.943 (0.909–0.976) | 0.922 (0.861–0.971) | 0.922 (0.858–0.971) | 0.780 |
| Pathological recall | 0.743 (0.421–1.000) | **0.971** (0.948–1.000) | **0.971** (0.948–1.000) | 0 |
| Pathological precision | 0.963 (0.800–1.000) | 0.680 (0.263–1.000) | 0.680 (0.263–1.000) | 0 |
| Suspect recall | 0.810 (0.644–0.948) | 0.690 (0.493–0.851) | 0.759 (0.600–0.902) | 0 |
| Suspect precision | 0.922 (0.826–1.000) | 1.000 (1.000–1.000) | 0.917 (0.796–1.000) | 0 |
| Mean cost (illustrative costs) | 0.272 | 0.137 | **0.128** | |

Intervals come from a bootstrap that resamples whole test recordings, since segments of one recording
are correlated.

What the numbers say:
- **The thresholds trade false alarms for missed cases.** Both threshold rules catch 34 of 35 Pathological
  segments instead of 26. The cost is 16 false Pathological flags, 14 of them on Normal segments, so about
  a third of their Pathological flags are wrong. Separately, calibration moves 16 Suspect segments to Normal,
  which is why Suspect recall drops under the 95%-recall threshold; the cost rule's Suspect threshold
  recovers part of that (Suspect recall 0.759 instead of 0.690).
- **Pathological numbers are uncertain.** The 35 Pathological test segments come from only 9 recordings,
  hence intervals like 0.42–1.00 for recall at the default operating point.
- **Accuracy is misleading here.** Predicting "Normal" for everything already gets 0.78, so models are
  selected and reported on macro-F1, which weighs the three classes equally.
- **Suspect is much better than before, but varies by split.** Suspect precision was 0.57 with the
  previous model (Random Forest + SMOTE, evaluated on a slightly different split). It is 0.92 on this
  split and 0.78 on average across the 10 splits (0.61–0.92). The comparison below shows SMOTE was not
  the best choice for most models. Why it varies is analysed [below](#why-suspect-precision-varies).

### Why Suspect precision varies

`scripts/repeated_splits.py` saves every test prediction with its recording
(`results/repeated_predictions.csv`), and the notebook's last section breaks the false Suspect calls
(segments predicted Suspect whose label is Normal or Pathological) down by recording. For the most likely
class:

- **The swings come from false Suspect calls.** Every test set has 58–59 Suspect segments and the model
  gets 39–51 of them right, but the false Suspect calls range from 4 to 32 per test set. Suspect precision
  tracks the number of false calls almost exactly (correlation −0.98).
- **A few recordings drive them.** The 139 false Suspect calls over the 10 test sets come from 32
  recordings; the top 5 give 60%. One recording, S8001041.dsp (17 segments, all labelled Normal), gives 50
  of them (36%): it is in 5 of the 10 test sets and 9–11 of its segments are called Suspect each time.
  Removing only the worst recording from each test set narrows Suspect precision from 0.61–0.92 to
  0.69–0.96.
- **They mostly do not come from mixed-label recordings.** 123 of the 352 recordings contain a
  Suspect-labelled segment. Only 27% of the false Suspect calls come from those recordings. Their non-Suspect
  segments are called Suspect twice as often (6.7% vs 3.3%), but they hold only 16% of the non-Suspect test
  segments. Most false calls are runs of segments from recordings never labelled Suspect at all.

The cost rule shows the same pattern (S8001041.dsp again gives the most false calls). The 95%-recall
threshold makes fewer Suspect calls and so fewer false ones (56 over the 10 test sets, against 139 for the
most likely class).

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

SMOTE and class weights both distort predicted probabilities. Three options compete on out-of-fold log
loss within the training set: the model's own probabilities ("none"), a sigmoid calibrator and an
isotonic calibrator, each fit on held-out recordings. On the notebook's split sigmoid wins (out-of-fold
log loss 0.254, against 0.279 for none and 0.321 for isotonic). **On the test set it made the probabilities
worse, though** (log loss 0.147 → 0.200, Brier 0.085 → 0.107). Across the 10 splits "none" never had the
lowest out-of-fold log loss (sigmoid won 8 times, isotonic twice), so the training data consistently favour
a calibrator even though it did not help on this test set. Both threshold rules work on the chosen
probabilities because they were chosen on them.

### Cost-based decisions

The cost rule needs a cost for every kind of error. `DEFAULT_COSTS` (in `src/fetal_health/decision.py`)
is **an illustrative assumption, not clinical guidance**: any false alarm costs 1, calling a Suspect
case Normal 2, calling a Pathological case Suspect 5 and calling it Normal 10. Real costs would have to
come from clinicians.

How much the choice matters, on the notebook's split: the cost of calling a Pathological case Normal is
varied (calling it Suspect costs half as much), and both thresholds are chosen again on out-of-fold
training predictions only.

| Pathological miss cost | Pathological threshold | Suspect threshold | Test Pathological recall | Test Pathological precision | Test macro-F1 | Pathological flags |
|---|---|---|---|---|---|---|
| 2 | 0.33 | 0.34 | 0.857 | 0.882 | 0.895 | 34 |
| 5 | 0.19 | 0.33 | 0.971 | 0.680 | 0.861 | 50 |
| 10 (default) | 0.19 | 0.33 | 0.971 | 0.680 | 0.861 | 50 |
| 20 | 0.19 | 0.33 | 0.971 | 0.680 | 0.861 | 50 |
| 50 | 0.03 | 0.34 | 1.000 | 0.422 | 0.754 | 83 |

Costs from 5 to 20 give the same rule; a very high miss cost catches the last Pathological segment at the
price of 32 more false alarms (48 instead of 16).

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
   resampled. The calibration and both decision rules are also chosen with cross-validation on the training set.
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
7. Choose the calibration (none, sigmoid or isotonic) with the lowest out-of-fold log loss on the
   training set. On the chosen out-of-fold probabilities, choose two decision rules: a Pathological
   threshold for 95% recall, and Pathological and Suspect thresholds that minimise an illustrative
   misclassification cost (with a sensitivity check over the cost of a missed Pathological case).
   Evaluate both on the test set.
8. Explain predictions with SHAP, and compare stratified, recording-grouped and proxy-grouped CV.
9. Repeat steps 4–7 on 10 different splits (`scripts/repeated_splits.py`) to see how much the results
   depend on the split, and break down the false Suspect calls by recording.

## Project structure

```
fetal_health_prediction.ipynb   # the full analysis, top to bottom
fetal_health.csv                # dataset (Kaggle)
recording_ids.csv               # source recording of each row (UCI release)
results/
    repeated_splits.csv         # one row per train/test split from scripts/repeated_splits.py
    repeated_predictions.csv    # one row per test segment and split: recording, label, predictions
scripts/
    build_recording_ids.py      # rebuilds recording_ids.csv from the UCI CTG.xls
    repeated_splits.py          # reruns selection and evaluation on 10 splits
src/fetal_health/
    data.py                     # loading, validation, deduplication, recording groups
    train.py                    # leak-free pipelines, imbalance strategies, holdout split, model search
    evaluate.py                 # per-class metrics, bootstrap intervals, CV protocol comparison
    decision.py                 # calibration choice, 95%-recall threshold, cost-based thresholds
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
if you don't already have it. A full run of the notebook took 1–6 minutes on a 32-core desktop; most of
it is the model search.

The notebook only reads `results/repeated_splits.csv` and `results/repeated_predictions.csv`. To recompute
them, run `python scripts/repeated_splits.py` (11 minutes for the 10 splits in the last run on a 32-core
machine; an earlier run took about half an hour). It saves each split as it finishes and skips splits
already in `repeated_splits.csv`, so delete both files first for a fresh run.

`recording_ids.csv` is already in the repository. To rebuild it, download `CTG.xls` from the
[UCI repository](https://archive.ics.uci.edu/dataset/193/cardiotocography), then run
`pip install xlrd` and `python scripts/build_recording_ids.py path/to/CTG.xls`.

## Limitations

- **One dataset.** There is no external validation. Each test set has only 8–10 recordings with
  Pathological segments, and macro-F1 moves by a standard deviation of 0.04 between splits. The bootstrap
  intervals and the repeated splits make this uncertainty visible, but they cannot shrink it.
- **Labels are expert judgements.** "Suspect" in particular is a borderline category on which clinicians
  often disagree, and 88 recordings have segments with different labels.
- **The chosen calibration can still hurt on new data.** "No calibration" now competes with sigmoid and
  isotonic, but on the notebook's split the training data still pick sigmoid, and it made the test log
  loss worse (0.147 → 0.200). Neither set of probabilities should be read as an exact risk.
- **The costs are illustrative.** The cost rule replaces a single 95%-recall target (met on 6 of 10 test
  sets) with an explicit trade-off between missed cases and false alarms, but the costs are an assumption.
  On the notebook's split, moving the Pathological miss cost from 2 to 50 moves the Pathological threshold
  from 0.33 to 0.03 and the Pathological flags from 34 to 83. Real costs must come from clinicians.
- **Suspect precision depends on a few recordings.** Calibration and both thresholds are now chosen per
  split, exactly as in the notebook, so the repeated splits run the full procedure. Their spread in
  Suspect precision is driven by a handful of recordings with many false Suspect calls (one of them gives
  36% of all false calls), so it will depend on which such recordings a new data set happens to contain.
- **SHAP shows associations, not causes.** The SHAP values explain the uncalibrated model in log-odds,
  not the calibrated probabilities.
