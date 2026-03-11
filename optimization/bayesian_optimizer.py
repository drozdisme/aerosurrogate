"""Bayesian Optimization for airfoil shape — AeroSurrogate v3.0."""
import logging, time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    best_value: float; best_geometry: Dict[str,float]
    best_params_raw: np.ndarray; objective: str
    n_trials: int; n_completed: int; elapsed_seconds: float
    pareto_front: Optional[List[Dict]] = None
    history: List[Dict] = field(default_factory=list)

class BayesianOptimizer:
    OBJECTIVES = ("max_lift_drag","min_drag","max_lift","pareto")

    def __init__(self, predictor, flow_conditions, objective="max_lift_drag",
                 parametrization="scalar", n_trials=100, sampler="tpe",
                 seed=42, constraints=None, timeout=None, use_botorch=False):
        if objective not in self.OBJECTIVES:
            raise ValueError(f"objective must be one of {self.OBJECTIVES}")
        self.predictor=predictor; self.flow=flow_conditions
        self.objective=objective; self.n_trials=n_trials
        self.sampler_name=sampler; self.seed=seed
        self.constraints=constraints or {}; self.timeout=timeout
        from optimization.airfoil_parametrization import PARAMETRIZATIONS
        self.param_space = PARAMETRIZATIONS[parametrization]()
        self._history: List[Dict] = []

    def _evaluate(self, geom):
        inp={**geom, **self.flow}
        r=self.predictor(inp)
        preds=r.get("predictions",r)
        return {k: v["value"] if isinstance(v,dict) else float(v) for k,v in preds.items()}

    def _obj_val(self, preds):
        Cl=preds.get("Cl",0.); Cd=max(preds.get("Cd",1e-3),1e-6)
        if self.objective=="max_lift_drag": return Cl/Cd
        if self.objective=="min_drag":     return -Cd
        if self.objective=="max_lift":     return Cl
        return Cl/Cd

    def _ok(self, preds):
        for k,v in self.constraints.items():
            fname=k[:-4] if k.endswith(("_min","_max")) else k
            if k.endswith("_min") and preds.get(fname,0)<v: return False
            if k.endswith("_max") and preds.get(fname,0)>v: return False
        return True

    def optimize(self) -> OptimizationResult:
        import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = (optuna.samplers.CmaEsSampler(seed=self.seed) if self.sampler_name=="cmaes"
               else optuna.samplers.RandomSampler(seed=self.seed) if self.sampler_name=="random"
               else optuna.samplers.TPESampler(seed=self.seed))
        t0=time.time()

        def _obj(trial):
            geom=self.param_space.suggest(trial)
            try: preds=self._evaluate(geom)
            except: return float("-inf")
            if not self._ok(preds): return float("-inf")
            val=self._obj_val(preds)
            self._history.append({"trial":trial.number,"value":val,"geometry":geom,"predictions":preds})
            return val

        study=optuna.create_study(direction="maximize",sampler=sampler)
        study.optimize(_obj,n_trials=self.n_trials,timeout=self.timeout,show_progress_bar=False)
        best=study.best_trial
        return OptimizationResult(
            best_value=best.value, best_geometry=best.params,
            best_params_raw=np.array(list(best.params.values())),
            objective=self.objective, n_trials=self.n_trials,
            n_completed=len(study.trials),
            elapsed_seconds=round(time.time()-t0,2), history=self._history)
