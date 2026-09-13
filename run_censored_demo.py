"""Run a small right-censored QAPF-EM demonstration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from qapf_em import (  # noqa: E402
    CensoredQAPFEM,
    GMMState,
    NumericalConfig,
    PenaltyConfig,
    QAPFEMConfig,
    censored_observed_log_likelihood,
)


def initialize_demo_state(Y: np.ndarray, n_components: int) -> GMMState:
    """Create deterministic, separated one-dimensional starting centres."""
    ordered = np.sort(Y[:, 0])
    indices = np.linspace(0, len(ordered) - 1, n_components + 2, dtype=int)[1:-1]
    means = ordered[indices, None]
    variance = float(np.var(Y[:, 0]) + 1e-4)
    covariances = np.full((n_components, 1, 1), variance)
    return GMMState(np.full(n_components, 1.0 / n_components), means, covariances)


def main() -> None:
    generator = np.random.default_rng(17)
    complete = np.concatenate((
        generator.normal(-1.8, 0.65, 120),
        generator.normal(1.7, 0.75, 120),
    ))[:, None]
    threshold = float(np.quantile(complete[:, 0], 0.70))
    delta = (complete[:, 0] <= threshold).astype(int)
    Y = complete.copy()
    Y[:, 0] = np.minimum(Y[:, 0], threshold)

    config = QAPFEMConfig(
        n_components_init=3,
        penalty=PenaltyConfig(lambda_weight=0.005, kappa_mcp=0.4),
        numerical=NumericalConfig(max_stages=250),
    )
    result = CensoredQAPFEM(config).fit(Y, delta, threshold, initialize_demo_state(Y, 3))
    likelihood = censored_observed_log_likelihood(Y, delta, threshold, result.state)

    print(f"censoring_rate={1 - delta.mean():.3f}")
    print(f"converged={result.converged}; retained_components={result.state.n_components}")
    print("weights=", np.round(result.state.weights, 4))
    print("means=", np.round(result.state.means.ravel(), 4))
    print(f"right_censored_log_likelihood={likelihood:.4f}")


if __name__ == "__main__":
    main()
