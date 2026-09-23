# Notebooks

`01_eda.ipynb` is the shared starting point: loading, label balance, missing
values, BRFSS special codes, correlations with the label, per-feature plots,
and the export of `feature_summary.csv`.

**Each of us copies it** as `eda_<name>.ipynb` (`eda_yahya.ipynb`,
`eda_sara.ipynb`, `eda_rania.ipynb`) and goes through the **whole** dataset:
all 321 columns, with the BRFSS 2015 codebook at hand.

What to bring to the meeting:

- your filled `feature_summary_<name>.csv`: for each column, its type (ordinal /
  nominal / continuous), the special codes, keep or drop, and a short note;
- the columns you would drop (almost empty, near-duplicates, identifiers);
- how you would fill the missing values, and which question blocks deserve a
  "was answered" flag;
- the transformations you believe matter, with the plot that convinced you.

At the meeting we merge the three versions into one pipeline, and only then
write it as a shared `.py` module.

Housekeeping: clear the outputs before committing (Kernel -> Restart & Clear
Output), and keep anything final out of the notebooks once the pipeline exists.
