"""论文中的完整观测、真实数据及第一维右删失数据构造。"""

from dataclasses import dataclass
from typing import Any

from qapf_em.state import GMMState


Array = Any


@dataclass(frozen=True)
class DatasetBundle:
    X: Array
    labels: Array
    reference_state: GMMState | None
    name: str


@dataclass(frozen=True)
class CensoredDatasetBundle:
    Y: Array
    delta: Array
    threshold: float
    labels: Array
    reference_state: GMMState | None
    name: str


def simulation_spec(dimension: int, model_index: int) -> GMMState:
    """从论文附录录入 1D/2D/3D、模型 1/2/3 的真实 GMM 参数。"""
    raise NotImplementedError


def sample_gaussian_mixture(state: GMMState, n_samples: int, seed: int) -> DatasetBundle:
    raise NotImplementedError


def load_real_dataset(name: str, standardize: bool = True) -> DatasetBundle:
    """加载并标准化 Iris、Wine 或 Seeds；由真实标签计算经验参数参照。"""
    raise NotImplementedError


def right_censor_first_coordinate(bundle: DatasetBundle, censoring_rate: float) -> CensoredDatasetBundle:
    """
    c 取第一维经验 (1-rate) 分位数；Y_1=min(X_1,c)，delta=I(X_1<=c)。
    """
    raise NotImplementedError


def semisynthetic_censoring(bundle: DatasetBundle, variable_index: int, censoring_rate: float) -> CensoredDatasetBundle:
    """
    将论文指定变量移到第一维后施加右删失：Iris 花瓣长度、Wine Proline、
    Seeds 籽粒长度；其余真实数据结构和类别标签保持不变。
    """
    raise NotImplementedError

