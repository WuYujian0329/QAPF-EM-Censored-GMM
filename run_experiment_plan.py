"""Outline of the thesis experiment plan.

The archive provides callable complete-observation and right-censored model
cores. The large repeated experiment loops remain explicit framework hooks so
that final manuscript settings can be added without hiding any decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from experiments.runners import (
    ExperimentPlan,
    run_ablation_study,
    run_censored_simulations,
    run_complete_simulations,
    run_penalty_sensitivity,
    run_real_data_experiments,
    run_semisynthetic_experiments,
)


def build_estimators():
    """Create the manuscript estimators after final experiment settings are fixed."""
    raise NotImplementedError


def main() -> None:
    """Document the scheduled experiments instead of silently running long jobs."""
    print("Experiment-plan scaffold: configure estimator factories before running the full 30-repeat study.")
    plan = ExperimentPlan(repeats=30, sample_size=500)
    estimators = build_estimators()

    # 第三章：完整观测实验
    run_complete_simulations(plan, estimators)
    run_real_data_experiments(plan, estimators)
    run_ablation_study(plan, estimators["QAPF-EM"])
    run_penalty_sensitivity(estimators["QAPF-EM"])

    # 第四章：第一维右删失扩展
    run_censored_simulations(plan, estimators)
    run_semisynthetic_experiments(plan, estimators)


if __name__ == "__main__":
    main()
