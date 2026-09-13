"""QAPF-EM 的混合比例惩罚与 MCP-LQA 接口。"""

from typing import Any

import numpy as np


Array = Any


def component_parameter_scale(dimension: int) -> float:
    """D_f = 1 + d + d(d+1)/2。"""
    return 1 + dimension + dimension * (dimension + 1) / 2


def total_free_parameters(n_components: int, dimension: int) -> int:
    """全协方差 GMM 的自由参数数目 K*D_f - 1。"""
    return int(n_components * component_parameter_scale(dimension) - 1)


def smooth_weight_penalty(weights: Array, n_samples: int, dimension: int, lambda_weight: float, epsilon: float) -> float:
    """n * lambda * D_f * sum(log(epsilon + pi_k) - log(epsilon))。"""
    weights = np.asarray(weights, dtype=float)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return float(n_samples * lambda_weight * component_parameter_scale(dimension) * np.sum(np.log(epsilon + weights) - np.log(epsilon)))


def penalized_weight_candidates(
    effective_counts: Array,
    n_samples: int,
    dimension: int,
    lambda_weight: float,
) -> Array:
    """计算 [(N_k/n)-lambda*D_f] / [1-K*lambda*D_f]。"""
    effective_counts = np.asarray(effective_counts, dtype=float)
    scale = component_parameter_scale(dimension)
    denominator = 1 - effective_counts.size * lambda_weight * scale
    if denominator <= 0:
        raise ValueError("weight penalty violates 1 - K*lambda*D_f > 0")
    return (effective_counts / n_samples - lambda_weight * scale) / denominator


def select_active_components(candidate_weights: Array, effective_counts: Array, epsilon_pi: float) -> Array:
    """截断为非负后筛选活动成分；若为空，保留 N_k 最大的一个。"""
    candidate_weights = np.asarray(candidate_weights, dtype=float)
    effective_counts = np.asarray(effective_counts, dtype=float)
    active = np.flatnonzero(candidate_weights > epsilon_pi)
    return active if active.size else np.asarray([int(np.argmax(effective_counts))])


def mcp_value(distance: float, n_samples: int, gamma_mcp: float, a_mcp: float) -> float:
    """论文中含 sqrt(n) 缩放的分段 MCP 函数。"""
    if distance < 0 or n_samples < 1 or gamma_mcp < 0 or a_mcp <= 1:
        raise ValueError("invalid MCP parameters")
    boundary = a_mcp * gamma_mcp / np.sqrt(n_samples)
    if distance <= boundary:
        return float(np.sqrt(n_samples) * gamma_mcp * distance - n_samples * distance**2 / (2 * a_mcp))
    return float(a_mcp * gamma_mcp**2 / 2)


def mcp_derivative(distance: float, n_samples: int, gamma_mcp: float, a_mcp: float) -> float:
    """sqrt(n) * max(gamma - sqrt(n)*distance/a, 0)。"""
    if distance < 0:
        raise ValueError("distance cannot be negative")
    return float(np.sqrt(n_samples) * max(gamma_mcp - np.sqrt(n_samples) * distance / a_mcp, 0.0))


def lqa_edge_weights(
    old_means: Array,
    n_samples: int,
    kappa_mcp: float,
    gamma_mcp: float,
    a_mcp: float,
    epsilon_eta: float,
) -> Array:
    """根据旧相邻均值距离计算 c_k = kappa*p'(eta_k)/max(eta_k, eps_eta)。"""
    old_means = np.asarray(old_means, dtype=float)
    if old_means.shape[0] < 2:
        return np.empty(0, dtype=float)
    distances = np.linalg.norm(np.diff(old_means, axis=0), axis=1)
    derivatives = np.asarray([mcp_derivative(distance, n_samples, gamma_mcp, a_mcp) for distance in distances])
    return kappa_mcp * derivatives / np.maximum(distances, epsilon_eta)


def solve_fused_means(
    provisional_centers: Array,
    effective_counts: Array,
    old_covariances: Array,
    edge_weights: Array,
) -> Array:
    """组装并直接求解论文中的分块三对角线性方程 B*mu=b。"""
    provisional_centers = np.asarray(provisional_centers, dtype=float)
    effective_counts = np.asarray(effective_counts, dtype=float)
    covariances = np.asarray(old_covariances, dtype=float)
    edge_weights = np.asarray(edge_weights, dtype=float)
    n_components, dimension = provisional_centers.shape
    if n_components == 1:
        return provisional_centers.copy()
    if edge_weights.size != n_components - 1:
        raise ValueError("edge_weights must have K-1 entries")
    system = np.zeros((n_components * dimension, n_components * dimension), dtype=float)
    rhs = np.zeros(n_components * dimension, dtype=float)
    for component in range(n_components):
        precision = np.linalg.pinv(covariances[component])
        diagonal_weight = (edge_weights[component - 1] if component else 0.0) + (edge_weights[component] if component < n_components - 1 else 0.0)
        block = effective_counts[component] * precision + diagonal_weight * np.eye(dimension)
        start = component * dimension
        system[start:start + dimension, start:start + dimension] = block
        rhs[start:start + dimension] = effective_counts[component] * precision @ provisional_centers[component]
        if component < n_components - 1:
            next_start = (component + 1) * dimension
            off_diagonal = -edge_weights[component] * np.eye(dimension)
            system[start:start + dimension, next_start:next_start + dimension] = off_diagonal
            system[next_start:next_start + dimension, start:start + dimension] = off_diagonal
    return np.linalg.solve(system, rhs).reshape(n_components, dimension)


def mcp_total_penalty(means: Array, n_samples: int, kappa_mcp: float, gamma_mcp: float, a_mcp: float) -> float:
    """kappa * sum_k MCP(||mu_(k+1)-mu_k||_2)。"""
    means = np.asarray(means, dtype=float)
    if means.shape[0] < 2:
        return 0.0
    distances = np.linalg.norm(np.diff(means, axis=0), axis=1)
    return float(kappa_mcp * sum(mcp_value(distance, n_samples, gamma_mcp, a_mcp) for distance in distances))
