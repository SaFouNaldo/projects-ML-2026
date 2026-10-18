# feature_selection

Exploration + lab journal. The **frozen pipeline code** lives at the root of the repository (not here).

## Contents (kept)

| File | Purpose |
|---|---|
| `selected_features.txt` | the 54 raw columns kept |
| `dropped_columns.csv` | funnel: dropped columns + reasons |
| `kept_columns.csv` | details of the 54 kept columns |
| `feature_selection.ipynb` | EDA / selection / F1 grid |
| `journal_experiences.tex` | history of experiments + F1 scores (for the team) |
| `univariate_auc_scan.csv` | univariate AUC scan (321 cols) |

## Frozen pipeline (repository root)

```text
preprocess.py  +  clinical_fe.py  +  interactions.py
        └────────────── pipeline.build_matrices ──────────────┘
                         → ~176 columns
```

- 54 raw columns → preprocess (~103)
- + 7 `NO_DUP_EXTRAS` clinical features
- + 66 pairwise interactions of the top-12 features

Recommended GD: `gamma=0.5`, `max_iters=3000`–`5000` + F1 threshold.
