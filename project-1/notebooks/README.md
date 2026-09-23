# Notebooks

| Notebook | Owner | What it is for |
|---|---|---|
| `01_eda.ipynb` | shared | First look at the data: label balance, missing values, special codes, correlations, per-feature plots. Run it once, do not commit the outputs. |
| `02_columns_<initials>.ipynb` | one per person | Copy `01_eda.ipynb` and keep only the sections you need. Go through YOUR third of the columns with the BRFSS 2015 codebook and fill in `feature_summary.csv`. |
| `03_models_<initials>.ipynb` | one per person | Try your model, record every cross-validation number you get. |

Rules:

- Notebooks are for exploring. Anything the final result depends on moves into
  `preprocessing.py`, `implementations.py`, `models_extra.py` or `evaluation.py`.
- Clear the outputs before committing (Kernel -> Restart & Clear Output), otherwise
  the diffs become unreadable.
- `feature_summary.csv` is written by the last cell of `01_eda.ipynb`. It is the
  shared table where each of us marks, for their columns: type (ordinal /
  nominal / continuous), the special codes, keep or drop, and notes.
