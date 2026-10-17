# Notebooks

## Exploration commune (conservée)

| Fichier | Rôle |
|---|---|
| `01_prepare_data.ipynb` | point de départ partagé |
| `02_cleaning_data.ipynb` | nettoyage / exploration équipe |
| `dropped_columns.csv` / `features_columns.csv` | listes issues de l’exploration commune |
| `brfss2015_rules_full.json` / `column_types.json` | références BRFSS |

## Sélection des variables — `feature_selection/`

Pipeline figé (preprocess + 7 extras `NO_DUP_EXTRAS`) :

| Fichier | Rôle |
|---|---|
| `feature_selection.ipynb` | EDA / sélection / grid F1 |
| `selected_features.txt` | 54 colonnes brutes retenues |
| `dropped_columns.csv` | colonnes écartées + raison |
| `kept_columns.csv` | détail des 54 gardées |

Code associé (dossier `project-1/`) : `preprocess.py`, `clinical_fe.py`, `pipeline.py`.

```python
from pathlib import Path
from pipeline import build_matrices

Xtr, y, Xte, meta = build_matrices(Path("data/dataset_unzipped/dataset"))
```
