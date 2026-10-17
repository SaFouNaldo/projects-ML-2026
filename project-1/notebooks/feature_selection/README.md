# feature_selection

Exploration + journal. Le **code pipeline figé** est dans `project-1/` (pas ici).

## Contenu (gardé)

| Fichier | Rôle |
|---|---|
| `selected_features.txt` | 54 colonnes brutes retenues |
| `dropped_columns.csv` | funnel : colonnes écartées + raisons |
| `kept_columns.csv` | détail des 54 gardées |
| `feature_selection.ipynb` | EDA / sélection / grid F1 |
| `journal_experiences.tex` | historique des essais + F1 (pour l’équipe) |
| `univariate_auc_scan.csv` | scan AUC univariée (321 cols) |

## Pipeline figé (`project-1/`)

```text
preprocess.py  +  clinical_fe.py  +  interactions.py
        └────────────── pipeline.build_matrices ──────────────┘
                         → ~176 colonnes
```

- 54 brutes → preprocess (~103)
- + 7 extras `NO_DUP_EXTRAS`
- + 66 interactions top-12

GD recommandé : `gamma=0.5`, `max_iters=3000`–`5000` + seuil F1.
