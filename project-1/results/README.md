# Results

`ablation.csv` is the single table all three of us write into. One row per
experiment, produced by `evaluation.run_ablation`:

| column | meaning |
|---|---|
| `name` | short description, e.g. "logreg + one-hot + flags" |
| `owner` | who ran it |
| `config` | the preprocessing switches used |
| `params` | lambda, gamma, iterations, class weight |
| `f1_mean`, `f1_std` | 5-fold cross-validation |
| `precision`, `recall`, `accuracy` | at the chosen threshold |
| `threshold` | the tuned decision threshold |
| `aicrowd_f1` | leaderboard score, when this run was submitted |

Always add a row rather than editing someone else's: the report's ablation
section is built from this file.
