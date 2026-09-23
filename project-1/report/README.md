# Report

- 2 pages maximum, LaTeX, IEEE format. References may go on a third page. No appendix.
- Start from `../latex-example-paper/latex-template.tex` (copy it here as `report.tex`).
- Figures are exported from the notebooks as PDF or PNG into `figures/`.
- Sections and owners:
  1. Introduction and problem -- A
  2. Data and preprocessing (missing values, special codes, encoding) -- B
  3. Models and training -- A
  4. Experiments: baselines, cross-validation, ablation -- C
  5. Discussion (threshold choice, recall vs precision in a medical setting) -- C
- Everything reported must be reproducible: give the lambda values, the step
  sizes, the number of folds, the exact feature transformations.
