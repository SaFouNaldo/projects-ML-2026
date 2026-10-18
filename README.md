# Heart attack prediction (MICHD) from the BRFSS survey

EPFL CS-433 Machine Learning, Fall 2026, Project 1. Team **F1-Cardiaque**:
Yahya Skalli, Sara El Harti, Rania El Kasmi.

The task is to predict whether a respondent of the 2015 BRFSS telephone survey has
had a heart attack or coronary heart disease (`MICHD`). The training set has
328 135 respondents and 321 columns (8.8 % positives), the test set 109 379
respondents. The metric is the F1 score of the positive class. Only Python, NumPy
and (for plots) matplotlib are used.

## Who did what

Each of us implemented two of the six required methods in `implementations.py`:
Rania `mean_squared_error_gd` and `mean_squared_error_sgd`, Sara `least_squares` and
`ridge_regression`, Yahya `logistic_regression` and `reg_logistic_regression`.
All three of us explored and preprocessed the full dataset and built models; the
code below is the pipeline we agreed on after comparing our results.

## Reproducing the submission

Put `dataset_to_release.zip` in `data/` (`run.py` unzips it), or put the CSV files in
`data/dataset_unzipped/dataset/` (`x_train.csv`, `y_train.csv`, `x_test.csv`).
Requirements: Python 3.9 or later and NumPy.

`run.py` has two modes. Both write `submissions/final_submission.csv`, the file of our
final AIcrowd submission, and both give exactly the same file.

**Inference (default, a few minutes).** Loads the trained models from
`models/gbm_final.npz` and predicts the test set. Only `x_test.csv` is read.

```
python run.py
```

**Training (a few hours on a laptop CPU).** Trains the four models from scratch with
5-fold cross-validation (about 15 000 trees), saves them in `models/gbm_final.npz`,
then writes the submission.

```
python run.py --mode train
```

The scores and trees of each fold are cached in `data/cache/run/` as soon as the
fold is done, so a training run that is stopped resumes where it was.

The final model is the average of four gradient-boosting models (the "members",
`CONFIGS` in `run.py`). They differ in the data preparation (`bins`), the number
of trees, the learning rate and the random seed; all use trees of depth 5:

| Member | Bins | Trees | Learning rate | Seed |
|---|---|---|---|---|
| `d5` | v1: raw BRFSS codes | 500 | 0.05 | 0 |
| `v2ctrl` | v2: "none" answers (88 / 888 / 8888) -> 0, alcohol in days per month | 500 | 0.05 | 0 |
| `v2s1` | v2 | 1000 | 0.03 | 1 |
| `v2s2` | v2 | 1000 | 0.03 | 2 |

Each training process uses one CPU core, so the members can be trained in
parallel, one per terminal, followed by one call that combines them (it only reads
the cache and takes a few minutes):

```
python run.py --mode train --configs d5
python run.py --mode train --configs v2ctrl
python run.py --mode train --configs v2s1
python run.py --mode train --configs v2s2
python run.py --mode train
```

What the training prints (laptop CPU, four members in parallel). The F1 values
are out-of-fold scores of 5-fold cross-validation:

| Member | Time per fold | F1 pooled | F1 nested | AUC |
|---|---|---|---|---|
| `d5` | about 26 min | 0.4457 | 0.4445 | 0.8691 |
| `v2ctrl` | about 26 min | 0.4453 | 0.4452 | 0.8692 |
| `v2s1` | about 53 min (1000 trees) | 0.4458 | 0.4451 | 0.8695 |
| `v2s2` | about 53 min (1000 trees) | 0.4455 | 0.4449 | 0.8695 |
| **average of the 4** | (combining: 18 s) | **0.4473** | **0.4463** | **0.8700** |

For example, the `d5` member prints

```
member {'tag': 'd5', 'bins': 'v1', 'trees': 500, 'lr': 0.05, 'seed': 0}
  d5: fold 1/5  1487s
  ...
  d5: fold 5/5  7809s
[d5] F1 pooled 0.4457  nested 0.4445  folds 0.4457±0.0015  AUC 0.8691  thr -1.2414
```

The combining call ends with

```
[average of 4 GBM] F1 pooled 0.4473  nested 0.4463  folds 0.4473±0.0017  AUC 0.8700  thr -1.2289
saved the models in models/gbm_final.npz
wrote submissions/final_submission.csv: threshold -1.2289, 12.24 % predicted positive
```

Inference (`python run.py`, about 5 minutes) writes a byte-identical file.
Everything is deterministic (fixed seeds), so training again gives the same models.

## Repository structure

```
run.py                 final model: average of 4 gradient-boosting models -> submission CSV
models/gbm_final.npz   the 20 trained fold models, binning and threshold used by run.py
implementations.py     the six methods required by the course
helpers.py             course helpers (CSV loading, submission writer)

preprocess.py          BRFSS cleaning and encoding for the linear models
clinical_fe.py         7 clinical features built from the raw answers
interactions.py        pairwise products of the 12 strongest columns
pipeline.py            preprocess + clinical features + interactions -> ~176 columns

research/              cross-validated experiments (see research/README.md)
  common.py            data loading and cache, folds, F1 threshold, pooled / nested F1
  gbm.py               histogram gradient boosting written in NumPy
  wide.py              generic encoding of all raw columns + Adam logistic regression
  cv_*.py              one script per experiment (six methods, logistic, wide, GBM)
  blend_weights.py, bag_equal.py   combinations of saved scores

notebooks/feature_selection/   selection of the columns (321 -> 54): notebook,
                               kept / dropped columns with the reason, AUC scan
  journal_experiences.tex      experiment log of the feature-engineering phase
                               (until 6 Oct): F1 of every variant tried on the
                               linear model, and why it was kept or rejected
latex-example-paper/   template of the report
data/                  not in git: zip, unzipped CSVs, caches
submissions/           not in git: CSV files written by run.py and research/
```

## Method

**Cleaning (`preprocess.py`).** The BRFSS answers are codes, not numbers. "Don't know"
and "refused" (7 / 9, 77 / 99, 777 / 999 depending on the question) become missing;
"none" (88) becomes 0; `ALCDAY5` mixes days per week (1xx) and days per month (2xx)
and is converted to days per month. Missing values are filled with the median
(numeric) or the mode (categorical), with a missing indicator for the 8 questions where
not answering is informative (skip patterns). Nominal columns are one-hot encoded,
fruit and vegetable sums get `log(1 + x)`, and everything is standardised. All the
statistics are computed on the training rows only.

**Feature selection (`notebooks/feature_selection/`).** From 321 to 54 columns:
identifiers, dates and survey weights removed, columns with more than 90 % missing
removed, one column of each redundant pair (correlation or Cramér's V >= 0.75)
removed, then a review of the remaining questions. `clinical_fe.py` adds 7 clinical
features and `interactions.py` the products of the 12 columns with the largest
logistic weight, refitted inside each fold.

**Evaluation (`research/common.py`).** Stratified 5-fold cross-validation, seed 42.
Each training row gets an out-of-fold (OOF) score. Since F1 depends on the decision
threshold, the threshold is chosen to maximise F1 on the OOF scores (pooled F1), and
also chosen on four folds and applied to the fifth (nested F1, the honest estimate).

**Final model (`run.py`).** Gradient-boosted trees written in NumPy
(`research/gbm.py`): the raw columns are cut into 63 quantile bins, each tree is grown
level by level with the Newton gain of the log-loss, and missing values go to the
better side of each split. Four models (depth 5, 500 to 1000 trees, two binnings,
three seeds) are trained with 5-fold cross-validation, the test score of each is the
mean of its 5 fold models, and the four logits are averaged with equal weights.

## Results (5-fold cross-validation)

| Model | nested F1 |
|---|---|
| Linear regression, SGD (`mean_squared_error_sgd`) | 0.4234 |
| Ridge regression (`ridge_regression`) | 0.4254 |
| Logistic regression (`logistic_regression`) | 0.4276 |
| Logistic regression on all raw columns, 2 207 encoded columns (`research/cv_wide.py`) | 0.4382 |
| Gradient boosting, one model (`d5`) | 0.4445 |
| **Average of 4 gradient-boosting models (`run.py`)** | **0.4463** |

The six methods are compared on the same matrix and protocol in
`research/cv_six_methods.py`; they all lie within 0.005 of each other. Choosing the
threshold matters more than the model: the logistic regression gets F1 0.21 with the
usual threshold 0.5 and 0.43 with the F1-optimal threshold (about 0.21).

**Label leak.** The question `HAREHAB1` (cardiac rehabilitation) is only asked to
people who reported a heart attack, so the 773 training rows where it is answered are
all positive. It is part of the provided data and the trees use it; without this
column the single gradient-boosting model gets 0.4356 (`research/cv_gbm.py v2noleak`).
