"""论文评价指标及标签对齐接口。"""

from dataclasses import dataclass
from typing import Any, Mapping

from qapf_em.state import GMMState


Array = Any


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    ari: float
    nll: float
    mse_weights: float
    mse_means: float
    mse_covariances: float


def hungarian_label_alignment(y_true: Array, y_pred: Array) -> Array:
    """用匈牙利算法消除聚类标签置换。"""
    raise NotImplementedError


def clustering_accuracy(y_true: Array, y_pred: Array) -> float:
    raise NotImplementedError


def adjusted_rand_index(y_true: Array, y_pred: Array) -> float:
    raise NotImplementedError


def align_component_parameters(reference: GMMState, estimate: GMMState) -> GMMState:
    """对混合权重、均值、协方差作同一最优成分匹配后再计算 MSE。"""
    raise NotImplementedError


def evaluate_complete_data(X: Array, y_true: Array, reference: GMMState, result: Any) -> Metrics:
    raise NotImplementedError


def evaluate_censored_data(bundle: Any, reference: GMMState, result: Any) -> Metrics:
    """NLL 必须使用统一的右删失观测似然，不能把阈值当真实值。"""
    raise NotImplementedError


def weighted_average_rank(metric_ranks: Mapping[str, float]) -> float:
    """
    [0.5*R_ACC + 0.5*R_ARI + R_NLL
     + (R_MSE_pi + R_MSE_mu + R_MSE_Sigma)/3] / 3。
    """
    return (
        0.5 * metric_ranks["accuracy"]
        + 0.5 * metric_ranks["ari"]
        + metric_ranks["nll"]
        + (
            metric_ranks["mse_weights"]
            + metric_ranks["mse_means"]
            + metric_ranks["mse_covariances"]
        ) / 3
    ) / 3

