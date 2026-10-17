# research/ -- models beyond the six course methods

NumPy only. Every experiment uses the same protocol: stratified 5-fold
cross-validation (seed 42), F1 threshold chosen on out-of-fold (OOF) scores,
F1 reported both *pooled* (one threshold on all OOF scores) and *nested*
(threshold of each fold chosen on the other four folds).
Test scores are the mean of the 5 fold models.

## Data and outputs

- Input: `project-1/data/dataset_unzipped/dataset/` (same as the rest of the repo).
- Cache (raw matrices, bins, OOF and test scores): `project-1/data/cache/research/`.
- Submissions: `project-1/submissions/research/`.

All three are ignored by git. Run the scripts from this folder.

## Files

| File | Role |
|---|---|
| `common.py` | data loading and cache, folds, F1, threshold search, pooled / nested report, AUC, score cache, CSV writer |
| `gbm.py` | histogram gradient boosting for the log-loss (quantile bins, level-wise trees, Newton gain, missing values sent left or right) |
| `wide.py` | generic encoding of all raw columns (one-hot, missing flags, quantile rank) + full-batch Adam logistic regression with L2 |
| `cv_logistic.py` | reference: base pipeline (176 columns) + IRLS logistic regression |
| `cv_six_methods.py` | the six functions of `implementations.py` on the same matrix and protocol |
| `cv_wide.py` | wide logistic regression, one run per `lambda` |
| `cv_gbm.py` | gradient boosting; scores saved every 100 trees |
| `blend_weights.py` | blend of saved scores with weights tuned on a grid, plus the `HAREHAB1` rule |
| `bag_equal.py` | equal-weight average of saved scores, threshold from OOF |

## Commands

```bash
cd project-1/research

python cv_logistic.py                    # ~2 min
python cv_six_methods.py                 # ~12 min
python cv_wide.py 1e-4 1e-3              # several hours

python cv_gbm.py d5       --trees 500  --depth 5 --lr 0.05             # bins v1
python cv_gbm.py v2ctrl   --trees 500  --depth 5 --lr 0.05 --bins v2
python cv_gbm.py v2s1     --trees 1000 --depth 5 --lr 0.03 --bins v2 --seed 1
python cv_gbm.py v2s2     --trees 1000 --depth 5 --lr 0.03 --bins v2 --seed 2
python cv_gbm.py v2noleak --trees 500  --depth 5 --lr 0.05 --bins v2 --drop HAREHAB1

python blend_weights.py C_gbm_d5 gbm_d5_t500
python bag_equal.py H_bag4 gbm_d5_t500 gbm_v2ctrl_t500 gbm_v2s1_t1000 gbm_v2s2_t1000
```

Each GBM run takes 30 to 60 minutes. `--bins v1` uses the raw codes;
`--bins v2` recodes the BRFSS "none" answers (88 / 888 / 8888) to 0 and adds
`ALCDAY5` converted to days per month.
