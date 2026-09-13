"""与论文符号一一对应的配置对象。"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnnealingConfig:
    beta0: float = 0.82
    gamma0: float = 0.10
    rho_beta: float = 1.018
    rho_gamma: float = 0.92


@dataclass(frozen=True)
class PenaltyConfig:
    lambda_weight: float = 4e-4
    kappa_mcp: float = 0.8
    gamma_mcp: float = 0.8
    a_mcp: float = 3.0
    smooth_epsilon: float = 1e-8
    simplify_components: bool = True


@dataclass(frozen=True)
class NumericalConfig:
    epsilon_pi: float = 1e-8
    epsilon_eta: float = 1e-8
    epsilon_gamma: float = 1e-8
    epsilon_objective: float = 1e-6
    epsilon_beta: float = 1e-14
    delta_sigma: float = 1e-6
    max_stages: int = 200


@dataclass(frozen=True)
class QAPFEMConfig:
    n_components_init: int
    annealing: AnnealingConfig = field(default_factory=AnnealingConfig)
    penalty: PenaltyConfig = field(default_factory=PenaltyConfig)
    numerical: NumericalConfig = field(default_factory=NumericalConfig)

    def validate(self, dimension: int) -> None:
        """检查论文中的参数可行域，不负责修正用户输入。"""
        component_scale = 1 + dimension + dimension * (dimension + 1) / 2
        denominator = 1 - self.n_components_init * self.penalty.lambda_weight * component_scale
        if denominator <= 0:
            raise ValueError("必须满足 1 - K_init * lambda * D_f > 0")
        if not (0 < self.annealing.beta0 <= 1):
            raise ValueError("beta0 必须位于 (0, 1]")
        if self.annealing.gamma0 < 0:
            raise ValueError("Gamma0 必须非负")
        if self.penalty.a_mcp <= 1:
            raise ValueError("MCP 形状参数 a 必须大于 1")
