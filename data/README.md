# Data files

`thesis_experiment_datasets.xlsx` is the English-language source workbook for
the benchmark and semi-synthetic right-censoring experiments.

It contains raw and standardized Iris, Wine, and Seeds data, plus three
right-censoring settings for each data set:

- `*_Raw`: original feature values with `target` and `class_name`.
- `*_Standardized`: feature-wise standardization; labels are unchanged.
- `*_Censored_10pct`, `*_Censored_30pct`, and `*_Censored_50pct`: the thesis
  feature is moved to the first coordinate and censored at the empirical 90%,
  70%, or 50% quantile.

In censored sheets, `delta = 1` means the first coordinate is observed and
`delta = 0` means its latent value is greater than `censor_threshold`.
