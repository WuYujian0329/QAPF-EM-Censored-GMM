"""完整观测高斯混合模型的基础数值接口。"""

from typing import Any

import numpy as np

from .state import GMMState


Array = Any


def _logsumexp(values: Array, axis: int | None = None) -> Array:
    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    total = np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    result = maximum + np.log(total)
    return np.squeeze(result, axis=axis) if axis is not None else float(result)


def _regularize_matrix(covariance: Array, floor: float = 1e-8) -> Array:
    covariance = np.asarray(covariance, dtype=float)
    covariance = (covariance + covariance.T) / 2
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues.min() < floor:
        covariance = covariance + np.eye(covariance.shape[0]) * (floor - eigenvalues.min())
    return covariance


def component_log_joint(X: Array, state: GMMState) -> Array:
    """返回 log[pi_k * phi_d(x_i; mu_k, Sigma_k)]，形状为 (n, K)。"""
    X = np.asarray(X, dtype=float)
    weights = np.asarray(state.weights, dtype=float)
    means = np.asarray(state.means, dtype=float)
    covariances = np.asarray(state.covariances, dtype=float)
    if X.ndim != 2 or means.ndim != 2 or X.shape[1] != means.shape[1]:
        raise ValueError("X and state means must have compatible two-dimensional shapes")
    n_samples, dimension = X.shape
    result = np.empty((n_samples, state.n_components), dtype=float)
    normalizer = dimension * np.log(2 * np.pi)
    for component in range(state.n_components):
        covariance = _regularize_matrix(covariances[component])
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise np.linalg.LinAlgError("covariance is not positive definite after regularization")
        centered = X - means[component]
        solved = np.linalg.solve(covariance, centered.T).T
        quadratic = np.sum(centered * solved, axis=1)
        result[:, component] = np.log(max(weights[component], np.finfo(float).tiny)) - 0.5 * (normalizer + logdet + quadratic)
    return result


def observed_log_likelihood(X: Array, state: GMMState) -> float:
    """以 log-sum-exp 计算经典观测数据对数似然。"""
    return float(np.sum(_logsumexp(component_log_joint(X, state), axis=1)))


def weighted_centers(X: Array, responsibilities: Array) -> tuple[Array, Array]:
    """返回 N_k 与暂时加权中心 x_bar_k。"""
    X = np.asarray(X, dtype=float)
    responsibilities = np.asarray(responsibilities, dtype=float)
    if responsibilities.shape[0] != X.shape[0]:
        raise ValueError("responsibilities and X must have the same number of rows")
    counts = responsibilities.sum(axis=0)
    centers = (responsibilities.T @ X) / np.maximum(counts[:, None], np.finfo(float).eps)
    return counts, centers


def regularized_covariances(
    X: Array,
    responsibilities: Array,
    means: Array,
    delta_sigma: float,
) -> Array:
    """加权协方差更新；先对称化，再加 delta_sigma * I_d。"""
    X = np.asarray(X, dtype=float)
    responsibilities = np.asarray(responsibilities, dtype=float)
    means = np.asarray(means, dtype=float)
    counts = responsibilities.sum(axis=0)
    dimension = X.shape[1]
    covariances = np.empty((means.shape[0], dimension, dimension), dtype=float)
    for component in range(means.shape[0]):
        centered = X - means[component]
        covariance = (centered * responsibilities[:, component, None]).T @ centered
        covariance /= max(counts[component], np.finfo(float).eps)
        covariances[component] = _regularize_matrix(covariance, delta_sigma)
    return covariances


def initialize_unfavorable_state(X: Array, n_components: int, seed: int) -> GMMState:
    """构造论文实验中供全部算法共享的同一组不良初始参数。"""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or n_components < 1:
        raise ValueError("X must be two-dimensional and n_components must be positive")
    generator = np.random.default_rng(seed)
    dimension = X.shape[1]
    global_mean = X.mean(axis=0)
    global_covariance = np.cov(X, rowvar=False, bias=True)
    if dimension == 1:
        global_covariance = np.asarray([[float(global_covariance)]])
    global_covariance = _regularize_matrix(global_covariance, 1e-6)
    scale = np.sqrt(np.diag(global_covariance))
    offsets = generator.normal(0, 0.15, size=(n_components, dimension)) * scale
    means = global_mean + offsets
    covariances = np.repeat((1.5 * global_covariance)[None, :, :], n_components, axis=0)
    return GMMState(np.full(n_components, 1 / n_components), means, covariances)
