"""论文中的 lambda-kappa 联合网格与 BIC+ 选择。"""

from dataclasses import replace
from typing import Any, Iterable

import numpy as np

from .config import QAPFEMConfig
from .penalties import component_parameter_scale, total_free_parameters
from .state import GMMState


Array = Any


def lambda_grid(n_components_init: int, dimension: int, stability_margin: float = 1e-6) -> list[float]:
    """[0, min(0.01,(1-margin)/(K_init*D_f))]，步长 0.0002。"""
    if n_components_init < 1 or dimension < 1:
        raise ValueError("n_components_init and dimension must be positive")
    maximum = min(0.01, (1 - stability_margin) / (n_components_init * component_parameter_scale(dimension)))
    return [round(value, 10) for value in np.arange(0.0, maximum + 1e-12, 0.0002)]


def kappa_grid() -> list[float]:
    """论文固定 gamma=0.8、a=3，仅搜索 kappa=0,0.1,...,2.0。"""
    return [round(0.1 * j, 10) for j in range(21)]


def bic_plus(log_likelihood: float, n_samples: int, n_components: int, dimension: int) -> float:
    p = total_free_parameters(n_components, dimension)
    return log_likelihood - 0.5 * p * __import__("math").log(n_samples)


def joint_bic_search(
    X: Array,
    initial_state: GMMState,
    base_config: QAPFEMConfig,
    lambdas: Iterable[float],
    kappas: Iterable[float],
):
    """
    对每个 (lambda,kappa) 从相同初始状态拟合 QAPF-EM，以最终有效成分数
    重新计算自由参数，并返回 BIC+ 最大的模型及完整调参记录。
    """
    from .model import QAPFEM
    from .gaussian import observed_log_likelihood

    def clone_state(state: GMMState) -> GMMState:
        return GMMState(np.asarray(state.weights, dtype=float).copy(), np.asarray(state.means, dtype=float).copy(), np.asarray(state.covariances, dtype=float).copy())

    records = []
    best = None
    for lambda_weight in lambdas:
        for kappa_mcp in kappas:
            penalty = replace(base_config.penalty, lambda_weight=float(lambda_weight), kappa_mcp=float(kappa_mcp))
            config = replace(base_config, penalty=penalty)
            result = QAPFEM(config).fit(X, clone_state(initial_state))
            likelihood = observed_log_likelihood(X, result.state)
            score = bic_plus(likelihood, X.shape[0], result.state.n_components, X.shape[1])
            record = {"lambda_weight": float(lambda_weight), "kappa_mcp": float(kappa_mcp), "bic_plus": score, "result": result}
            records.append(record)
            if best is None or score > best["bic_plus"]:
                best = record
    return best, records
