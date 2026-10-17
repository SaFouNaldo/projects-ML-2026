# Project 1 -- Heart attack prediction (MICHD)

Team **F1-Cardiaque** -- Yahya Skalli, Sara El Harti, Rania El Kasmi
EPFL CS-433 Machine Learning, Fall 2026. Deadline: 29 October 2026, 16:00.

## Getting the data

The data is **not** in the repository. Put `dataset_to_release.zip` in `data/`
and let the notebook unzip it, or unzip it by hand:

```
project-1/data/dataset_unzipped/dataset/{x_train.csv, y_train.csv, x_test.csv, sample-submission.csv}
```

`data/` and `data/cache/` are ignored by git.

## Current stage

1. **The six functions.** Implemented / merged into `implementations.py`.
2. **Sara FE (this branch).** Pipeline figé :
   - `preprocess_sara.py` + `notebooks/research_sara/features_sara.txt` (54 brutes)
   - `clinical_fe.py` — 7 extras `NO_DUP_EXTRAS` (sans redondance)
   - `interactions_sara.py` — pairwise top-12
   - `pipeline_sara.py` — entrée unique (`build_matrices`) → ~176 cols
3. **Models.** Logistic GD : $\gamma=0.5$, `max_iters` 3000–5000 + seuil F1 (OOF ~0.426).

## Who does what

| Functions in `implementations.py` | Owner |
|---|---|
| `mean_squared_error_gd`, `mean_squared_error_sgd` (+ MSE helpers) | Rania |
| `least_squares`, `ridge_regression` | Sara |
| `logistic_regression`, `reg_logistic_regression` (+ sigmoid, logistic helpers) | Yahya |

Everyone reviews someone else's pair before it is merged.

For the data, there is no split: each of us does the whole dataset, then we
compare at the meeting.

## Layout

| File | Contents |
|---|---|
| `implementations.py` | the six required methods (graded, auto-imported) |
| `helpers.py` | course helpers, do not modify |
| `preprocess_sara.py` | preprocess Sara (fit train → apply test) |
| `clinical_fe.py` | 7 extras cliniques sans doublon |
| `pipeline_sara.py` | Sara + extras → matrices |
| `notebooks/` | exploration équipe + `research_sara/` (pipeline Sara) |
| `research/` | modèles au-delà des six méthodes : GBM NumPy, logistique large, mélanges (voir `research/README.md`) |
| `latex-example-paper/` | the LaTeX template for the report |
| `submissions/` | CSV AICrowd (ignorés par git) |

## Conventions (agreed on, do not change alone)

- Labels are `{-1, 1}` everywhere, converted to `{0, 1}` only inside the logistic methods.
- Features: `tx` has shape `(N, D)` with the bias column already added.
- Weights are 1-D arrays of shape `(D,)`, never `(D, 1)`.
- Every function returns `(w, loss)` with the LAST `w`, and the loss of the
  penalized methods excludes the penalty term.
- Statistics (mean, std, medians, categories) are always computed on the training
  part only, then applied to validation and test.
- Seed is `42`.
- The metric is **F1**; accuracy is reported alongside but never used to decide.

## Checks before pushing

```
black .                  # the public tests check the formatting
```

Then, from the course repository's `grading_tests` folder:

```
pytest --github_link <repo-url-or-local-path> .
```

Note: those tests expect `README.md`, `implementations.py` and `run.py` at the
**root of the repository**, so we decide before the deadline whether to flatten
this folder or use a separate repo for the submission.

## Where things stand

- [ ] Rania: linear regression (GD, SGD)
- [ ] Sara: least squares, ridge
- [ ] Yahya: logistic, regularized logistic
- [ ] everyone: full exploration + preprocessing notebook
- [ ] meeting: agree on one shared preprocessing pipeline
