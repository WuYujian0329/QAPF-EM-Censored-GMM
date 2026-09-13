"""模型状态及可追踪的拟合结果。"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Array = Any


@dataclass
class GMMState:
    weights: Array       # shape: (K,)
    means: Array         # shape: (K, d)
    covariances: Array   # shape: (K, d, d)

    @property
    def n_components(self) -> int:
        return len(self.weights)

    def subset(self, active: Array) -> "GMMState":
        """同步保留活动成分。"""
        active = np.asarray(active)
        if active.dtype == bool:
            active = np.flatnonzero(active)
        if active.ndim != 1 or active.size == 0:
            raise ValueError("active must select at least one component")
        weights = np.asarray(self.weights, dtype=float)[active]
        weights = weights / weights.sum()
        return GMMState(
            weights=weights.copy(),
            means=np.asarray(self.means, dtype=float)[active].copy(),
            covariances=np.asarray(self.covariances, dtype=float)[active].copy(),
        )

    def reorder(self, order: Array) -> "GMMState":
        """按同一排列同步重排权重、均值和协方差。"""
        order = np.asarray(order, dtype=int)
        if order.ndim != 1 or order.size != self.n_components:
            raise ValueError("order must be a permutation of all components")
        if set(order.tolist()) != set(range(self.n_components)):
            raise ValueError("order must be a valid component permutation")
        return GMMState(
            weights=np.asarray(self.weights, dtype=float)[order].copy(),
            means=np.asarray(self.means, dtype=float)[order].copy(),
            covariances=np.asarray(self.covariances, dtype=float)[order].copy(),
        )


@dataclass
class IterationRecord:
    stage: int
    beta: float
    quantum_gamma: float
    n_components: int
    objective: float
    objective_change: float


@dataclass
class FitResult:
    state: GMMState
    responsibilities: Array
    labels: Array
    converged: bool
    history: list[IterationRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
