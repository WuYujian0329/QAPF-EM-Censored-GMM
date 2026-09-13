"""论文对照算法的统一接口；此文件不实现具体数值细节。"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .config import AnnealingConfig, NumericalConfig, PenaltyConfig, QAPFEMConfig
from .gaussian import observed_log_likelihood
from .model import QAPFEM
from .state import FitResult, GMMState, IterationRecord


Array = Any


class MixtureEstimator(Protocol):
    def fit(self, X: Array, initial_state: GMMState) -> FitResult: ...


@dataclass
class ClassicalEM:
    """CEM：beta=1、Gamma=0、无双惩罚。"""
    max_iter: int = 500

    def fit(self, X: Array, initial_state: GMMState) -> FitResult:
        config = QAPFEMConfig(
            n_components_init=initial_state.n_components,
            annealing=AnnealingConfig(beta0=1.0, gamma0=0.0, rho_beta=1.0, rho_gamma=0.0),
            penalty=PenaltyConfig(lambda_weight=0.0, kappa_mcp=0.0, simplify_components=False),
            numerical=NumericalConfig(max_stages=self.max_iter),
        )
        return QAPFEM(config).fit(X, initial_state)


@dataclass
class DAEM:
    """确定性退火 EM：Gamma=0，beta 逐步增至 1。"""
    beta0: float = 0.1
    rho_beta: float = 1.05
    max_iter: int = 500

    def fit(self, X: Array, initial_state: GMMState) -> FitResult:
        config = QAPFEMConfig(
            n_components_init=initial_state.n_components,
            annealing=AnnealingConfig(beta0=self.beta0, gamma0=0.0, rho_beta=self.rho_beta, rho_gamma=0.0),
            penalty=PenaltyConfig(lambda_weight=0.0, kappa_mcp=0.0, simplify_components=False),
            numerical=NumericalConfig(max_stages=self.max_iter),
        )
        return QAPFEM(config).fit(X, initial_state)


@dataclass
class SEM:
    """随机 EM：按后验责任度抽样潜在类别。"""
    seed: int = 0
    max_iter: int = 500

    def fit(self, X: Array, initial_state: GMMState) -> FitResult:
        return _stochastic_em(X, initial_state, self.seed, self.max_iter, stochastic_approximation=False)


@dataclass
class SAEM:
    """随机逼近 EM：维护随机充分统计量的递推平均。"""
    seed: int = 0
    max_iter: int = 500
    warmup: int = 20

    def fit(self, X: Array, initial_state: GMMState) -> FitResult:
        return _stochastic_em(X, initial_state, self.seed, self.max_iter, stochastic_approximation=True, warmup=self.warmup)


def _stochastic_em(
    X: Array,
    initial_state: GMMState,
    seed: int,
    max_iter: int,
    stochastic_approximation: bool,
    warmup: int = 20,
) -> FitResult:
    """Run SEM or SAEM with sampled labels and the common covariance-stabilized M-step."""
    X = np.asarray(X, dtype=float)
    helper = QAPFEM(
        QAPFEMConfig(
            n_components_init=initial_state.n_components,
            annealing=AnnealingConfig(beta0=1.0, gamma0=0.0, rho_beta=1.0, rho_gamma=0.0),
            penalty=PenaltyConfig(lambda_weight=0.0, kappa_mcp=0.0, simplify_components=False),
            numerical=NumericalConfig(max_stages=max_iter),
        )
    )
    generator = np.random.default_rng(seed)
    state = initial_state
    running_responsibilities = None
    history = []
    previous_objective = -np.inf
    converged = False
    responsibilities = None
    for iteration in range(max_iter):
        posterior = np.exp(helper._component_log_joint(X, state))
        posterior /= posterior.sum(axis=1, keepdims=True)
        draws = np.asarray([generator.choice(state.n_components, p=row) for row in posterior])
        sampled = np.eye(state.n_components)[draws]
        if stochastic_approximation:
            rate = 1.0 if iteration < warmup else (iteration - warmup + 2) ** -0.6
            running_responsibilities = sampled if running_responsibilities is None else (1 - rate) * running_responsibilities + rate * sampled
            responsibilities = running_responsibilities
        else:
            responsibilities = sampled
        state, responsibilities = helper._penalized_m_step(X, responsibilities, state)
        objective = observed_log_likelihood(X, state)
        change = abs(objective - previous_objective)
        history.append(IterationRecord(iteration, 1.0, 0.0, state.n_components, objective, change))
        if iteration > 0 and change < 1e-6:
            converged = True
            break
        previous_objective = objective
    return FitResult(state, responsibilities, np.argmax(responsibilities, axis=1), converged, history)
