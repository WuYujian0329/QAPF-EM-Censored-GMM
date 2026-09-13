"""DQAEM/QAPF-EM 共用的量子 E 步。"""

from typing import Any

import numpy as np


Array = Any


def _softmax(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def cyclic_noncommuting_operator(n_components: int) -> Array:
    """
    构造论文中的循环非对易算符：
    K=1 时为 0；K=2 时为 S；K>=3 时为 S + S^H。
    """
    if n_components < 1:
        raise ValueError("n_components must be positive")
    operator = np.zeros((n_components, n_components), dtype=float)
    if n_components == 1:
        return operator
    if n_components == 2:
        operator[0, 1] = operator[1, 0] = 1.0
        return operator
    for index in range(n_components):
        operator[index, (index + 1) % n_components] = 1.0
        operator[(index + 1) % n_components, index] = 1.0
    return operator


def total_hamiltonian(component_energies: Array, quantum_gamma: float) -> Array:
    """对单个样本构造 diag(h_i1,...,h_iK) + Gamma * z_nc。"""
    component_energies = np.asarray(component_energies, dtype=float)
    if component_energies.ndim != 1:
        raise ValueError("component_energies must be one-dimensional")
    return np.diag(component_energies) + quantum_gamma * cyclic_noncommuting_operator(component_energies.size)


def quantum_responsibilities(
    component_log_joint: Array,
    beta: float,
    quantum_gamma: float,
) -> Array:
    """
    对每个样本进行 Hermitian 特征分解，按最小特征值平移后计算谱形式责任度。

    Gamma=0 时应直接走 DAEM 的幂后验分支；Gamma=0 且 beta=1 时应等于
    经典 EM 后验责任度。
    """
    component_log_joint = np.asarray(component_log_joint, dtype=float)
    if component_log_joint.ndim != 2 or not (0 < beta <= 1) or quantum_gamma < 0:
        raise ValueError("log-joint values must be two-dimensional, beta in (0, 1], and gamma nonnegative")
    if quantum_gamma == 0:
        return _softmax(beta * component_log_joint)
    responsibilities = np.empty_like(component_log_joint)
    for index, log_joint in enumerate(component_log_joint):
        eigenvalues, eigenvectors = np.linalg.eigh(total_hamiltonian(-log_joint, quantum_gamma))
        spectrum = np.exp(-beta * (eigenvalues - eigenvalues.min()))
        density = (eigenvectors * spectrum) @ eigenvectors.T / spectrum.sum()
        diagonal = np.maximum(np.real(np.diag(density)), 0.0)
        responsibilities[index] = diagonal / diagonal.sum()
    return responsibilities


def quantum_negative_free_energy(
    component_log_joint: Array,
    beta: float,
    quantum_gamma: float,
) -> float:
    """由各样本局部配分函数计算未惩罚量子负自由能。"""
    component_log_joint = np.asarray(component_log_joint, dtype=float)
    if quantum_gamma == 0:
        scaled = beta * component_log_joint
        maximum = scaled.max(axis=1, keepdims=True)
        log_partition = maximum[:, 0] + np.log(np.exp(scaled - maximum).sum(axis=1))
    else:
        log_partition = []
        for log_joint in component_log_joint:
            eigenvalues = np.linalg.eigvalsh(total_hamiltonian(-log_joint, quantum_gamma))
            shifted = -beta * (eigenvalues - eigenvalues.min())
            log_partition.append(-beta * eigenvalues.min() + np.log(np.exp(shifted).sum()))
        log_partition = np.asarray(log_partition)
    return float(np.sum(log_partition) / beta)
