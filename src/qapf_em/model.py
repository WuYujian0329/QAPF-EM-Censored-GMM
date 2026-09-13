"""完整观测 QAPF-EM 的主体流程。"""

from math import inf
from typing import Any

import numpy as np

from .config import QAPFEMConfig
from .gaussian import component_log_joint, regularized_covariances, weighted_centers
from .penalties import (
    lqa_edge_weights,
    mcp_total_penalty,
    penalized_weight_candidates,
    select_active_components,
    solve_fused_means,
    smooth_weight_penalty,
)
from .quantum import quantum_negative_free_energy, quantum_responsibilities
from .state import FitResult, GMMState, IterationRecord


Array = Any


class QAPFEM:
    """Complete-observation QAPF-EM estimator used in the main article."""

    def __init__(self, config: QAPFEMConfig):
        self.config = config

    def fit(self, X: Array, initial_state: GMMState) -> FitResult:
        n_samples, dimension = X.shape
        self.config.validate(dimension)

        state = initial_state
        beta = self.config.annealing.beta0
        quantum_gamma = self.config.annealing.gamma0
        previous_objective = -inf
        history: list[IterationRecord] = []
        converged = False
        responsibilities = None

        for stage in range(self.config.numerical.max_stages):
            log_joint = self._component_log_joint(X, state)
            responsibilities = quantum_responsibilities(log_joint, beta, quantum_gamma)
            state, responsibilities = self._penalized_m_step(X, responsibilities, state)

            objective = self._penalized_objective(X, state, beta, quantum_gamma)
            change = abs(objective - previous_objective)
            history.append(IterationRecord(stage, beta, quantum_gamma, state.n_components, objective, change))

            converged = self._should_stop(beta, quantum_gamma, change)
            if converged:
                break

            previous_objective = objective
            beta = min(1.0, self.config.annealing.rho_beta * beta)
            quantum_gamma = self.config.annealing.rho_gamma * quantum_gamma

        labels = self._labels_from_responsibilities(responsibilities)
        return FitResult(state, responsibilities, labels, converged, history)

    def _component_log_joint(self, X: Array, state: GMMState) -> Array:
        """Return complete-observation component log-joint values."""
        return component_log_joint(X, state)

    def _penalized_m_step(self, X: Array, responsibilities: Array, old_state: GMMState) -> tuple[GMMState, Array]:
        cfg = self.config
        n_samples, dimension = X.shape
        effective_counts, provisional_centers = weighted_centers(X, responsibilities)

        if cfg.penalty.simplify_components:
            candidate = penalized_weight_candidates(
                effective_counts, n_samples, dimension, cfg.penalty.lambda_weight
            )
            active = select_active_components(candidate, effective_counts, cfg.numerical.epsilon_pi)
            old_state, responsibilities, candidate = self._prune(old_state, responsibilities, candidate, active)
            effective_counts, provisional_centers = weighted_centers(X, responsibilities)
        else:
            candidate = effective_counts / n_samples

        order = self._stable_first_coordinate_order(provisional_centers)
        old_state, responsibilities, candidate = self._reorder(old_state, responsibilities, candidate, order)
        effective_counts, provisional_centers = weighted_centers(X, responsibilities)
        weights = candidate / candidate.sum()

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
        covariances = regularized_covariances(
            X, responsibilities, means, cfg.numerical.delta_sigma
        )
        return GMMState(weights, means, covariances), responsibilities

    def _penalized_objective(self, X: Array, state: GMMState, beta: float, quantum_gamma: float) -> float:
        """量子负自由能 - 混合比例惩罚 - MCP 相邻均值距离惩罚。"""
        cfg = self.config
        n_samples, dimension = X.shape
        log_joint = self._component_log_joint(X, state)
        free_energy = quantum_negative_free_energy(log_joint, beta, quantum_gamma)
        weight_penalty = smooth_weight_penalty(
            state.weights, n_samples, dimension, cfg.penalty.lambda_weight, cfg.penalty.smooth_epsilon
        ) if cfg.penalty.simplify_components else 0.0
        fusion_penalty = mcp_total_penalty(
            state.means, n_samples, cfg.penalty.kappa_mcp, cfg.penalty.gamma_mcp, cfg.penalty.a_mcp
        )
        return free_energy - weight_penalty - fusion_penalty

    def _prune(self, state: GMMState, responsibilities: Array, weights: Array, active: Array):
        """同步删除参数和责任度列；此处不对剩余责任度逐行归一化。"""
        active = np.asarray(active, dtype=int)
        pruned_state = state.subset(active)
        pruned_responsibilities = np.asarray(responsibilities, dtype=float)[:, active]
        pruned_weights = np.maximum(np.asarray(weights, dtype=float)[active], 0.0)
        if pruned_weights.sum() <= 0:
            pruned_weights = pruned_state.weights.copy()
        return pruned_state, pruned_responsibilities, pruned_weights

    def _stable_first_coordinate_order(self, provisional_centers: Array) -> Array:
        """按暂时加权中心第一坐标稳定升序排列。"""
        return np.argsort(np.asarray(provisional_centers, dtype=float)[:, 0], kind="stable")

    def _reorder(self, state: GMMState, responsibilities: Array, weights: Array, order: Array):
        """同步重排参数、责任度列与候选权重。"""
        order = np.asarray(order, dtype=int)
        return state.reorder(order), np.asarray(responsibilities, dtype=float)[:, order], np.asarray(weights, dtype=float)[order]

    def _should_stop(self, beta: float, quantum_gamma: float, objective_change: float) -> bool:
        num = self.config.numerical
        return (
            beta >= 1 - num.epsilon_beta
            and quantum_gamma <= num.epsilon_gamma
            and objective_change < num.epsilon_objective
        )

    def _labels_from_responsibilities(self, responsibilities: Array) -> Array:
        if responsibilities is None:
            raise RuntimeError("fit did not complete an E-step")
        return np.argmax(np.asarray(responsibilities), axis=1)
