"""与论文第三、四章实验结构一致的任务编排。"""

from dataclasses import dataclass
from typing import Any, Callable, Mapping


EstimatorFactory = Callable[..., Any]


@dataclass(frozen=True)
class ExperimentPlan:
    repeats: int = 30
    sample_size: int = 500
    dimensions: tuple[int, ...] = (1, 2, 3)
    model_indices: tuple[int, ...] = (1, 2, 3)
    censoring_rates: tuple[float, ...] = (0.10, 0.30, 0.50)


def run_complete_simulations(plan: ExperimentPlan, estimators: Mapping[str, EstimatorFactory]):
    """1D/2D/3D × 模型1/2/3；全部方法共享每次重复的不良初始值。"""
    raise NotImplementedError


def run_real_data_experiments(plan: ExperimentPlan, estimators: Mapping[str, EstimatorFactory]):
    """标准化后的 Iris、Wine、Seeds；确定性与随机算法均纳入 30 次汇总。"""
    raise NotImplementedError


def run_ablation_study(plan: ExperimentPlan, qapf_factory: EstimatorFactory):
    """
    在三个维度的模型2比较：Full QAPF-EM、DQAEM+MCP、
    DQAEM+weight penalty、DQAEM、DAEM；报告 ACCURACY 与 ARI。
    """
    raise NotImplementedError


def run_penalty_sensitivity(qapf_factory: EstimatorFactory):
    """一维模型2、K_init=7；围绕 BIC+ 选值分别改变 lambda 与 kappa。"""
    raise NotImplementedError


def run_censored_simulations(plan: ExperimentPlan, estimators: Mapping[str, EstimatorFactory]):
    """
    1D/2D/3D 模型2，n=500，固定 K=4；按 10%/30%/50% 对第一维右删失。
    """
    raise NotImplementedError


def run_semisynthetic_experiments(plan: ExperimentPlan, estimators: Mapping[str, EstimatorFactory]):
    """Iris、Wine、Seeds 的指定变量右删失，保留原类别标签。"""
    raise NotImplementedError


def summarize_mean_std_and_rank(raw_results: Any):
    """输出均值±标准差、逐指标名次、综合平均名次和最优次数。"""
    raise NotImplementedError

