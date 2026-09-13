# Right-censored QAPF-EM

This repository implements the thesis extension in which only the first
coordinate can be right-censored. For each observation,

```text
Y[i, 0] = min(X[i, 0], c)
delta[i] = 1 if X[i, 0] <= c, otherwise 0
Y[i, 1:] = X[i, 1:]
```

`delta = 0` does not mean that the first coordinate equals the threshold. It
means that the latent value exceeds it.

## Component likelihood

For an observed first coordinate, the implementation uses the usual full
Gaussian density. For a censored first coordinate, it uses the marginal
density of the observed remaining coordinates multiplied by the conditional
normal survival probability:

```text
pi[k] * phi_{d-1}(Y[i, 1:]) * Phi-bar((c - m[i, k]) / s[k])
```

The helper `conditional_normal_parameters` computes `m[i, k]` and `s[k]^2`.
`stable_mills_ratio` evaluates `phi(a) / Phi-bar(a)` in the right tail without
naively dividing two underflow-prone floating-point values.

## M-step sufficient statistics

For a censored observation, the module uses the truncated-normal moments

```text
E[X1 | X1 > c]  = m + s * M(a)
E[X1^2 | X1 > c] = m^2 + 2*m*s*M(a) + s^2*(1 + a*M(a))
```

and combines them with the observed remaining coordinates to form
`E[X | observed, Z=k]` and `E[X X^T | observed, Z=k]`. The existing weight
penalty, component deletion, stable ordering, MCP-LQA mean fusion, covariance
regularization, and annealing schedule are then reused unchanged.

## Public interface

```python
from qapf_em import CensoredQAPFEM

result = CensoredQAPFEM(config).fit(Y, delta, threshold, initial_state)
```

`censored_observed_log_likelihood(Y, delta, threshold, state)` calculates the
matching observed-data likelihood. Do not evaluate censored data with the
ordinary complete-observation likelihood, because replacing a censored value by
the threshold discards the survival information.
