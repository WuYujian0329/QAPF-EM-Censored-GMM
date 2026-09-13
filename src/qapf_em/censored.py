"""Right-censored first-coordinate extension of QAPF-EM.

For a censored observation, ``delta[i] == 0`` means that the latent first
coordinate exceeds ``threshold``. The remaining coordinates are observed.
The implementation follows the likelihood and truncated-normal sufficient
statistics specified in Chapter 6 of the accompanying thesis.
"""

from __future__ import annotations

from math import erfc, sqrt
from typing import Any

import numpy as np

from .model import QAPFEM
from .penalties import (
    lqa_edge_weights,
    penalized_weight_candidates,
    select_active_components,
    solve_fused_means,
)
from .state import FitResult, GMMState


Array = Any
_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def _logsumexp(values: Array, axis: int) -> Array:
    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    total = np.exp(values - maximum).sum(axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(total), axis=axis)


def _regularize(covariance: Array, floor: float = 1e-8) -> Array:
    covariance = np.asarray(covariance, dtype=float)
    covariance = (covariance + covariance.T) / 2.0
    minimum = float(np.linalg.eigvalsh(covariance).min())
    if minimum < floor:
        covariance = covariance + np.eye(covariance.shape[0]) * (floor - minimum)
    return covariance


def _log_normal_survival(values: Array) -> Array:
    """Compute log(Phi-bar(values)) without right-tail underflow."""
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    central = values < 8.0
    if np.any(central):
        survival = np.vectorize(erfc, otypes=[float])(values[central] / sqrt(2.0)) / 2.0
        result[central] = np.log(survival)
    if np.any(~central):
        tail = values[~central]
        correction = 1.0 - tail ** -2 + 3.0 * tail ** -4
        result[~central] = -0.5 * tail**2 - np.log(tail) - _LOG_SQRT_2PI + np.log(correction)
    return result


def _log_multivariate_normal(values: Array, mean: Array, covariance: Array) -> Array:
    values = np.asarray(values, dtype=float)
    mean = np.asarray(mean, dtype=float)
    covariance = _regularize(covariance)
    centered = values - mean
    solved = np.linalg.solve(covariance, centered.T).T
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise np.linalg.LinAlgError("marginal covariance is not positive definite")
    quadratic = np.sum(centered * solved, axis=1)
    return -0.5 * (mean.size * np.log(2.0 * np.pi) + logdet + quadratic)


def _validate_censoring_data(Y: Array, delta: Array, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    Y = np.asarray(Y, dtype=float)
    delta = np.asarray(delta)
    valid_delta = delta.ndim == 1 and delta.shape[0] == Y.shape[0] and np.all(np.isin(delta, (0, 1, False, True)))
    if Y.ndim != 2 or Y.shape[0] == 0 or not valid_delta:
        raise ValueError("Y must be non-empty two-dimensional data and delta must be aligned 0/1 values")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return Y, delta.astype(bool)


class CensoredQAPFEM(QAPFEM):
    """QAPF-EM for Gaussian mixtures with right censoring in coordinate one."""

    def fit(self, Y: Array, delta: Array, threshold: float, initial_state: GMMState) -> FitResult:
        Y, delta = _validate_censoring_data(Y, delta, threshold)
        if np.asarray(initial_state.means).shape[1] != Y.shape[1]:
            raise ValueError("Y and initial_state must have the same feature dimension")
        self._delta = delta
        self._threshold = float(threshold)
        result = super().fit(Y, initial_state)
        result.metadata.update({"right_censored": True, "threshold": self._threshold})
        return result

    def _component_log_joint(self, Y: Array, state: GMMState) -> Array:
        """Use full densities when observed and marginal density times survival when censored."""
        from .gaussian import component_log_joint

        Y, delta = _validate_censoring_data(Y, self._delta, self._threshold)
        log_joint = component_log_joint(Y, state)
        censored = ~delta
        if not np.any(censored):
            return log_joint

        means = np.asarray(state.means, dtype=float)
        covariances = np.asarray(state.covariances, dtype=float)
        weights = np.asarray(state.weights, dtype=float)
        conditional_means, conditional_variances = conditional_normal_parameters(Y, state)
        for component in range(state.n_components):
            if Y.shape[1] == 1:
                marginal_log_density = np.zeros(censored.sum(), dtype=float)
            else:
                marginal_log_density = _log_multivariate_normal(
                    Y[censored, 1:], means[component, 1:], covariances[component, 1:, 1:]
                )
            standardized_threshold = (
                self._threshold - conditional_means[censored, component]
            ) / np.sqrt(conditional_variances[component])
            log_joint[censored, component] = (
                np.log(max(weights[component], np.finfo(float).tiny))
                + marginal_log_density
                + _log_normal_survival(standardized_threshold)
            )
        return log_joint

    def _penalized_m_step(self, Y: Array, responsibilities: Array, old_state: GMMState) -> tuple[GMMState, Array]:
        """Use censored conditional moments in the shared deletion and fusion M-step."""
        cfg = self.config
        Y, _ = _validate_censoring_data(Y, self._delta, self._threshold)
        responsibilities = np.asarray(responsibilities, dtype=float)
        n_samples, dimension = Y.shape
        moments, second_moments = censored_conditional_moments(
            Y, self._delta, self._threshold, old_state
        )
        effective_counts, provisional_centers = _weighted_centers_from_moments(responsibilities, moments)

        if cfg.penalty.simplify_components:
            candidate = penalized_weight_candidates(
                effective_counts, n_samples, dimension, cfg.penalty.lambda_weight
            )
            active = select_active_components(candidate, effective_counts, cfg.numerical.epsilon_pi)
            old_state, responsibilities, candidate = self._prune(old_state, responsibilities, candidate, active)
            moments = moments[:, active, :]
            second_moments = second_moments[:, active, :, :]
            effective_counts, provisional_centers = _weighted_centers_from_moments(responsibilities, moments)
        else:
            candidate = effective_counts / n_samples

        order = self._stable_first_coordinate_order(provisional_centers)
        old_state, responsibilities, candidate = self._reorder(old_state, responsibilities, candidate, order)
        moments = moments[:, order, :]
        second_moments = second_moments[:, order, :, :]
        effective_counts, provisional_centers = _weighted_centers_from_moments(responsibilities, moments)
        weights = candidate / max(candidate.sum(), np.finfo(float).eps)

        edge_weights = lqa_edge_weights(
            old_state.means,
            n_samples,
            cfg.penalty.kappa_mcp,
            cfg.penalty.gamma_mcp,
            cfg.penalty.a_mcp,
            cfg.numerical.epsilon_eta,
        )
        means = solve_fused_means(
            provisional_centers, effective_counts, old_state.covariances, edge_weights
        )
        covariances = _covariances_from_censored_moments(
            responsibilities, second_moments, means, cfg.numerical.delta_sigma
        )
        return GMMState(weights, means, covariances), responsibilities


def conditional_normal_parameters(Y: Array, state: GMMState) -> tuple[Array, Array]:
    """Return m_ik and s_k^2 for X_1 conditional on the remaining coordinates."""
    Y = np.asarray(Y, dtype=float)
    means = np.asarray(state.means, dtype=float)
    covariances = np.asarray(state.covariances, dtype=float)
    n_samples, dimension = Y.shape
    if means.ndim != 2 or means.shape[1] != dimension:
        raise ValueError("Y and state means must have matching dimensions")
    conditional_means = np.empty((n_samples, state.n_components), dtype=float)
    conditional_variances = np.empty(state.n_components, dtype=float)
    for component in range(state.n_components):
        covariance = _regularize(covariances[component])
        if dimension == 1:
            conditional_means[:, component] = means[component, 0]
            conditional_variances[component] = covariance[0, 0]
            continue
        rest_covariance = _regularize(covariance[1:, 1:])
        coefficient = np.linalg.solve(rest_covariance, covariance[1:, 0])
        conditional_means[:, component] = means[component, 0] + (Y[:, 1:] - means[component, 1:]) @ coefficient
        conditional_variances[component] = covariance[0, 0] - covariance[0, 1:] @ coefficient
    return conditional_means, np.maximum(conditional_variances, np.finfo(float).eps)


def stable_mills_ratio(a: Array) -> Array:
    """Compute phi(a)/Phi-bar(a) from a stable log-survival calculation."""
    a = np.asarray(a, dtype=float)
    log_density = -0.5 * a**2 - _LOG_SQRT_2PI
    return np.exp(np.minimum(log_density - _log_normal_survival(a), 700.0))


def censored_conditional_moments(
    Y: Array,
    delta: Array,
    threshold: float,
    state: GMMState,
) -> tuple[Array, Array]:
    """Return E[X_i|obs,Z=k] and E[X_i X_i'|obs,Z=k] for every i and k."""
    Y, delta = _validate_censoring_data(Y, delta, threshold)
    n_components = state.n_components
    first = np.repeat(Y[:, None, :], n_components, axis=1)
    second = np.einsum("nid,nie->nide", first, first)
    censored = ~delta
    if not np.any(censored):
        return first, second

    conditional_means, conditional_variances = conditional_normal_parameters(Y, state)
    for component in range(n_components):
        standard_deviation = np.sqrt(conditional_variances[component])
        mean = conditional_means[censored, component]
        standardized_threshold = (threshold - mean) / standard_deviation
        mills = stable_mills_ratio(standardized_threshold)
        first_coordinate = mean + standard_deviation * mills
        first_coordinate_square = (
            mean**2
            + 2.0 * mean * standard_deviation * mills
            + conditional_variances[component] * (1.0 + standardized_threshold * mills)
        )
        first[censored, component, 0] = first_coordinate
        second[censored, component, 0, 0] = first_coordinate_square
        if Y.shape[1] > 1:
            observed_rest = Y[censored, 1:]
            cross = first_coordinate[:, None] * observed_rest
            second[censored, component, 0, 1:] = cross
            second[censored, component, 1:, 0] = cross
            second[censored, component, 1:, 1:] = np.einsum("ni,nj->nij", observed_rest, observed_rest)
    return first, second


def censored_observed_log_likelihood(Y: Array, delta: Array, threshold: float, state: GMMState) -> float:
    """Return the observed-data log likelihood under the right-censoring mechanism."""
    estimator = CensoredQAPFEM.__new__(CensoredQAPFEM)
    estimator._delta = np.asarray(delta)
    estimator._threshold = float(threshold)
    log_joint = estimator._component_log_joint(Y, state)
    return float(np.sum(_logsumexp(log_joint, axis=1)))


def _weighted_centers_from_moments(responsibilities: Array, moments: Array) -> tuple[Array, Array]:
    responsibilities = np.asarray(responsibilities, dtype=float)
    moments = np.asarray(moments, dtype=float)
    counts = responsibilities.sum(axis=0)
    centers = np.einsum("nk,nkd->kd", responsibilities, moments) / np.maximum(counts[:, None], np.finfo(float).eps)
    return counts, centers


def _covariances_from_censored_moments(
    responsibilities: Array,
    second_moments: Array,
    means: Array,
    delta_sigma: float,
) -> Array:
    responsibilities = np.asarray(responsibilities, dtype=float)
    second_moments = np.asarray(second_moments, dtype=float)
    means = np.asarray(means, dtype=float)
    n_components, dimension = means.shape
    counts = responsibilities.sum(axis=0)
    covariances = np.empty((n_components, dimension, dimension), dtype=float)
    for component in range(n_components):
        expectation = np.einsum(
            "n,nij->ij", responsibilities[:, component], second_moments[:, component]
        ) / max(counts[component], np.finfo(float).eps)
        covariance = expectation - np.outer(means[component], means[component])
        covariances[component] = _regularize(covariance, delta_sigma)
    return covariances
