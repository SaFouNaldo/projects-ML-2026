# Project 1 -- Heart attack prediction (MICHD)

Team **Saraya** -- Yahya Skalli, Sara El Harti, Rania El Kasmi
EPFL CS-433 Machine Learning, Fall 2026. Deadline: 29 October 2026, 16:00.

## Getting the data

The data is **not** in the repository. Put `dataset_to_release.zip` in `data/`
and let the notebook unzip it, or unzip it by hand:

```
project-1/data/dataset_unzipped/dataset/{x_train.csv, y_train.csv, x_test.csv, sample-submission.csv}
```

`data/`, `data/cache/` and `submissions/` are ignored by git.

## Layout

| File | Contents | Owner |
|---|---|---|
| `implementations.py` | the six required methods (graded, auto-imported) | A: GD + SGD, B: least squares + ridge, C: logistic + regularized logistic |
| `preprocessing.py` | loading, cleaning, missing values, encoding, scaling, feature engineering | all three, one third of the columns each |
| `evaluation.py` | metrics, cross-validation, threshold and hyperparameter search, ablation | C |
| `models_extra.py` | the extra models: weighted logistic regression, boosting, forest, network, ensemble | A / B / C |
| `run.py` | reproduces our best submission exactly | A |
| `helpers.py` | course helpers, do not modify | -- |
| `notebooks/` | exploration, see `notebooks/README.md` | all three |
| `results/ablation.csv` | every experiment we ran | all three |
| `report/` | the 2-page LaTeX report | C leads |

## Conventions (agreed on, do not change alone)

- Labels are `{-1, 1}` everywhere, converted to `{0, 1}` only inside the logistic methods.
- Features: `tx` has shape `(N, D)` with the bias column already added.
- Statistics (mean, std, medians, categories) are always computed on the training
  fold only, then applied to validation and test.
- Seed is `42`, folds come from `evaluation.build_k_indices`, 5 folds.
- The metric is **F1**; accuracy is reported alongside but never used to decide.
- The decision threshold is tuned on each validation fold, never on the test set.
- Resampling, if used, is applied to the training fold only.

## Workflow

```
git checkout -b <name>/<topic>      # never commit to main directly
black .                             # the grading tests check the formatting
pytest --github_link . ../ML_course/projects/project1/grading_tests   # signature tests
```

Open a pull request, one teammate reviews, then merge.

## Where things stand

- [ ] the six functions implemented and passing the public tests
- [ ] `feature_summary.csv` filled in for all 321 columns
- [ ] baseline logistic regression submitted to AICrowd
- [ ] ablation table started
- [ ] report outline written
