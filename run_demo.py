"""Run a small complete-observation QAPF-EM demonstration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from qapf_em import GMMState, NumericalConfig, PenaltyConfig, QAPFEM, QAPFEMConfig  # noqa: E402


def initialize_demo_state(X: np.ndarray, n_components: int) -> GMMState:
    """Use deterministic, separated quantile centres for this deletion sanity check."""
    ordered = X[np.argsort(X[:, 0], kind="stable")]
    indices = np.linspace(0, len(ordered) - 1, n_components + 2, dtype=int)[1:-1]
    means = ordered[indices]
    covariance = np.cov(X, rowvar=False, bias=True) + np.eye(X.shape[1]) * 1e-6
    covariances = np.repeat(covariance[None, :, :], n_components, axis=0)
    return GMMState(np.full(n_components, 1.0 / n_components), means, covariances)


def main() -> None:
    generator = np.random.default_rng(7)
    true_weights = np.array([0.5, 0.5])
    true_means = np.array([[-2.0, 0.0], [2.0, 0.5]])
    X = np.vstack((generator.normal(true_means[0], 0.8, size=(100, 2)), generator.normal(true_means[1], 0.9, size=(100, 2))))
    # Start from four components although the data were generated from two.
    # This demonstration-only lambda makes the weight-deletion step observable.
    initial_state = initialize_demo_state(X, n_components=4)
    demo_config = QAPFEMConfig(
        n_components_init=4,
        penalty=PenaltyConfig(lambda_weight=0.01, kappa_mcp=0.8),
        numerical=NumericalConfig(max_stages=1000),
    )
    result = QAPFEM(demo_config).fit(X, initial_state)
    print(f"converged={result.converged}; retained_components={result.state.n_components}")
    print("weights=", np.round(result.state.weights, 4))
    print("means=\n", np.round(result.state.means, 4))
    print("covariances=\n", np.round(result.state.covariances, 4))
    print("true_weights=", true_weights)
    print("true_means=\n", true_means)


if __name__ == "__main__":
    main()
