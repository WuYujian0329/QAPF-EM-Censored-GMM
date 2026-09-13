"""Small numerical checks for the right-censored core."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qapf_em import (  # noqa: E402
    CensoredQAPFEM,
    GMMState,
    NumericalConfig,
    PenaltyConfig,
    QAPFEMConfig,
    censored_conditional_moments,
    censored_observed_log_likelihood,
)
from qapf_em.config import AnnealingConfig  # noqa: E402


class CensoredQAPFEMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.standard_normal = GMMState(
            weights=np.array([1.0]),
            means=np.array([[0.0]]),
            covariances=np.array([[[1.0]]]),
        )

    def test_truncated_standard_normal_moments(self) -> None:
        first, second = censored_conditional_moments(
            np.array([[0.0]]), np.array([0]), 0.0, self.standard_normal
        )
        self.assertAlmostEqual(first[0, 0, 0], np.sqrt(2.0 / np.pi), places=8)
        self.assertAlmostEqual(second[0, 0, 0, 0], 1.0, places=8)

    def test_right_censored_likelihood_uses_survival_probability(self) -> None:
        value = censored_observed_log_likelihood(
            np.array([[0.0]]), np.array([0]), 0.0, self.standard_normal
        )
        self.assertAlmostEqual(value, np.log(0.5), places=8)

    def test_two_dimensional_censoring_conditions_on_observed_coordinate(self) -> None:
        state = GMMState(
            weights=np.array([1.0]),
            means=np.array([[0.0, 0.0]]),
            covariances=np.array([[[1.0, 0.5], [0.5, 1.0]]]),
        )
        first, second = censored_conditional_moments(
            np.array([[0.0, 0.0]]), np.array([0]), 0.0, state
        )
        expected_variance = 0.75
        self.assertAlmostEqual(first[0, 0, 0], np.sqrt(expected_variance * 2.0 / np.pi), places=8)
        self.assertAlmostEqual(second[0, 0, 0, 0], expected_variance, places=8)
        self.assertAlmostEqual(second[0, 0, 0, 1], 0.0, places=8)
        expected_likelihood = np.log(0.5) - 0.5 * np.log(2.0 * np.pi)
        self.assertAlmostEqual(
            censored_observed_log_likelihood(np.array([[0.0, 0.0]]), np.array([0]), 0.0, state),
            expected_likelihood,
            places=8,
        )

    def test_small_fit_preserves_responsibility_shape(self) -> None:
        complete = np.array([[-1.2], [-0.8], [-0.4], [0.1], [0.8], [1.4]])
        threshold = 0.5
        delta = (complete[:, 0] <= threshold).astype(int)
        Y = np.minimum(complete, threshold)
        initial = GMMState(
            weights=np.array([0.5, 0.5]),
            means=np.array([[-0.7], [0.7]]),
            covariances=np.array([[[1.0]], [[1.0]]]),
        )
        config = QAPFEMConfig(
            n_components_init=2,
            annealing=AnnealingConfig(beta0=1.0, gamma0=0.0, rho_beta=1.0, rho_gamma=0.0),
            penalty=PenaltyConfig(lambda_weight=0.0, kappa_mcp=0.0, simplify_components=False),
            numerical=NumericalConfig(max_stages=4),
        )
        result = CensoredQAPFEM(config).fit(Y, delta, threshold, initial)
        self.assertEqual(result.responsibilities.shape, (6, 2))
        self.assertTrue(np.allclose(result.responsibilities.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
