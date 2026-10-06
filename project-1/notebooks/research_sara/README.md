# research_sara

Travail Sara (feature engineering + preprocess + extras cliniques).

| Fichier | Rôle |
|---|---|
| `notebook_sara.ipynb` | EDA, funnel de sélection, grid F1 |
| `features_sara.txt` | 54 colonnes brutes retenues |
| `dropped_columns_sara.csv` | ce qui a été droppé et pourquoi |
| `features_columns_sara.csv` | détail des 54 gardées |

Pipeline code (dans `project-1/`) :

- `preprocess_sara.py`
- `clinical_fe.py` — 7 extras `NO_DUP_EXTRAS`
- `pipeline_sara.py` — `build_matrices(...)`
