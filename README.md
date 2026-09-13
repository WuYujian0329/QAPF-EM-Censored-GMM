# QAPF-EM reproducibility framework with right censoring

This repository accompanies the thesis *Gaussian-mixture parameter estimation
based on the QAPF-EM algorithm*. It is organized as a GitHub-ready source
archive and contains both the complete-observation QAPF-EM core and the
right-censored first-coordinate extension described in the thesis.

## Repository contents

```text
.
├── data/                         English benchmark and censored-data workbook
├── docs/right_censoring.md       Model, likelihood, and API documentation
├── src/qapf_em/                  Complete and right-censored estimators
├── src/experiments/              Thesis experiment-plan framework
├── tests/test_censored.py        Numerical checks for censored sufficient statistics
├── run_demo.py                   Complete-observation demonstration
├── run_censored_demo.py          Right-censored demonstration
├── run_experiment_plan.py        Explicit outline of the long experiment suite
├── validate_archive.py           Dependency-free release-layout validation
├── requirements.txt
└── CITATION.cff
```

## Installation and quick checks

```bash
pip install -r requirements.txt
python validate_archive.py
python run_demo.py
python run_censored_demo.py
python -m unittest tests.test_censored
```

`run_demo.py` starts from four components for two-component synthetic data and
checks the complete-observation deletion flow. `run_censored_demo.py` applies
right censoring to the first coordinate of synthetic one-dimensional data,
fits `CensoredQAPFEM`, and reports the matching censored likelihood.

## Right-censored extension

The extension assumes that only coordinate one can be censored:

```text
Y[i, 0] = min(X[i, 0], threshold)
delta[i] = 1  if X[i, 0] <= threshold
delta[i] = 0  if X[i, 0] > threshold
```

For censored records, the E-step replaces the full Gaussian density with the
marginal density of the observed coordinates times a conditional normal
survival probability. The M-step restores first and second sufficient moments
with the truncated-normal Mills ratio. Component deletion, stable sorting,
MCP-LQA mean fusion, covariance regularization, and the annealing schedule are
shared with the complete-observation implementation. See
[`docs/right_censoring.md`](docs/right_censoring.md) for formulas and usage.

```python
from qapf_em import CensoredQAPFEM

result = CensoredQAPFEM(config).fit(Y, delta, threshold, initial_state)
```

## Data workbook

`data/thesis_experiment_datasets.xlsx` is entirely in English. It contains
raw and standardized Iris, Wine, and Seeds data, plus the 10%, 30%, and 50%
semi-synthetic right-censoring settings. The README inside `data/` defines all
fields and sheet names.

## Experiment scope

The estimator cores and their small demonstrations are callable. The modules
under `src/experiments/` deliberately retain explicit `NotImplementedError`
hooks for the manuscript-scale 30-repeat simulation, real-data, ablation,
sensitivity, and semi-synthetic result pipelines. This keeps every unresolved
experimental choice visible rather than claiming an exact regeneration of all
thesis tables from a compact release archive.

## Before a public GitHub release

1. Add the authors' chosen open-source license.
2. Complete the experiment factories and fixed-seed result generators.
3. Compare regenerated tables against the thesis values using documented
   numerical tolerances.
4. Add a release tag and archive the tagged version with a DOI service if a
   permanent citation is needed.

## Citation

Please cite the accompanying thesis or article. Update `CITATION.cff` with the
final bibliographic record and DOI after publication.
