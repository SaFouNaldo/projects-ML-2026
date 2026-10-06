# Notebooks

## Exploration commune (conservée)

| Fichier | Rôle |
|---|---|
| `01_prepare_data.ipynb` | point de départ partagé |
| `02_cleaning_data.ipynb` | nettoyage / exploration équipe |
| `dropped_columns.csv` / `features_columns.csv` | listes issues de l’exploration commune |
| `brfss2015_rules_full.json` / `column_types.json` | références BRFSS |

## Recherche Sara — `research_sara/`

Pipeline figé (Sara + 7 extras `NO_DUP_EXTRAS`) :

| Fichier | Rôle |
|---|---|
| `notebook_sara.ipynb` | EDA / sélection / grid F1 |
| `features_sara.txt` | 54 colonnes brutes retenues |
| `dropped_columns_sara.csv` | colonnes écartées + raison |
| `features_columns_sara.csv` | détail des 54 gardées |

Code associé (dossier `project-1/`) : `preprocess_sara.py`, `clinical_fe.py`, `pipeline_sara.py`.

```python
from pathlib import Path
from pipeline_sara import build_matrices

Xtr, y, Xte, meta = build_matrices(Path("data/dataset_unzipped/dataset"))
```
