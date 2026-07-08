from __future__ import annotations

import optuna

from icframe.domain.evaluation import EvaluationResult
from icframe.domain.mutations import OptimizationResult, SearchSpace, TrialOutcome
from icframe.domain.scenario import Scenario
from icframe.pipelines.evaluate import evaluate_trace
from icframe.pipelines.simulate import run_simulation
from icframe.ports.analyzer import AnalyzerPort
from icframe.ports.optimizer import OptimizerPort
from icframe.ports.simulator import SimulatorPort
from icframe.ports.solver import SolverPort


class OptunaOptimizer(OptimizerPort):
    def __init__(
        self,
        solver: SolverPort,
        simulator: SimulatorPort,
        analyzer: AnalyzerPort,
        seed: int = 0,
    ) -> None:
        self.solver = solver
        self.simulator = simulator
        self.analyzer = analyzer
        self.seed = seed
        self.last_evaluation: EvaluationResult | None = None

    def optimize(
        self,
        scenario: Scenario,
        search_space: SearchSpace,
        trials: int,
    ) -> OptimizationResult:
        outcomes: list[TrialOutcome] = []

        def objective(trial: optuna.Trial) -> float:
            mutated = scenario.model_copy(deep=True)
            params: dict[str, float] = {}
            for mutation in search_space.float_params:
                value = trial.suggest_float(mutation.name, mutation.low, mutation.high)
                self._apply_float_param(mutated, mutation.name, value)
                params[mutation.name] = value

            trace, _ = run_simulation(
                scenario=mutated,
                solver=self.solver,
                simulator=self.simulator,
                analyzer=self.analyzer,
                seed=self.seed,
            )
            if trace.graph is None:
                raise ValueError("simulation trace is missing graph projection")
            evaluation = evaluate_trace(
                mutated,
                trace,
                self.analyzer.summarize(trace.graph),
            )
            self.last_evaluation = evaluation
            trial.set_user_attr("visible_score", evaluation.visible_score)
            outcomes.append(
                TrialOutcome(
                    number=trial.number,
                    params=params,
                    visible_score=evaluation.visible_score,
                    trusted_score=evaluation.trusted_score,
                )
            )
            return evaluation.trusted_score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            study_name=f"{scenario.name}-trusted-search",
        )
        study.optimize(objective, n_trials=trials)
        return OptimizationResult(
            study_name=study.study_name,
            best_params=study.best_params,
            best_value=study.best_value,
            trials=outcomes,
        )

    @staticmethod
    def _apply_float_param(scenario: Scenario, name: str, value: float) -> None:
        if hasattr(scenario.incentives, name):
            setattr(scenario.incentives, name, value)
            return
        if hasattr(scenario.simulation, name):
            setattr(scenario.simulation, name, value)
            return
        raise ValueError(f"unsupported optimization parameter: {name}")
