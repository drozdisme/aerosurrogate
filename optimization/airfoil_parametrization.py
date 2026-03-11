"""Airfoil shape parametrizations for Bayesian optimization."""
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class ParamBounds:
    name: str; low: float; high: float; log_scale: bool = False

class ScalarAirfoilParam:
    PARAMS = [
        ParamBounds("thickness_ratio",0.06,0.22), ParamBounds("camber",0.00,0.08),
        ParamBounds("camber_position",0.25,0.65), ParamBounds("leading_edge_radius",0.002,0.04),
        ParamBounds("trailing_edge_angle",8.,24.), ParamBounds("aspect_ratio",4.,14.),
        ParamBounds("taper_ratio",0.2,0.9),        ParamBounds("sweep_angle",0.,40.),
        ParamBounds("twist_angle",-3.,3.),         ParamBounds("dihedral_angle",0.,7.),
    ]
    def suggest(self, trial) -> Dict[str,float]:
        return {p.name: trial.suggest_float(p.name,p.low,p.high,log=p.log_scale) for p in self.PARAMS}
    @property
    def dim(self): return len(self.PARAMS)

class NACAFourDigitParam:
    PARAMS = [
        ParamBounds("naca_m",0.,0.09), ParamBounds("naca_p",0.2,0.7),
        ParamBounds("naca_t",0.06,0.22), ParamBounds("aspect_ratio",4.,14.),
        ParamBounds("taper_ratio",0.2,0.9), ParamBounds("sweep_angle",0.,40.),
    ]
    def suggest(self, trial) -> Dict[str,float]:
        r={p.name:trial.suggest_float(p.name,p.low,p.high) for p in self.PARAMS}
        m,p,t=r["naca_m"],r["naca_p"],r["naca_t"]
        return dict(thickness_ratio=t,camber=m,camber_position=p,
            leading_edge_radius=0.5*t**2,trailing_edge_angle=12+20*t,
            aspect_ratio=r["aspect_ratio"],taper_ratio=r["taper_ratio"],
            sweep_angle=r["sweep_angle"],twist_angle=0.,dihedral_angle=3.)

class CSTParam:
    PARAMS = [
        ParamBounds("thickness_ratio",0.06,0.22), ParamBounds("camber",0.,0.10),
        ParamBounds("camber_position",0.25,0.65), ParamBounds("aspect_ratio",4.,14.),
        ParamBounds("taper_ratio",0.2,0.9),        ParamBounds("sweep_angle",0.,40.),
    ]
    def suggest(self, trial) -> Dict[str,float]:
        r={p.name:trial.suggest_float(p.name,p.low,p.high) for p in self.PARAMS}
        t=r["thickness_ratio"]
        return dict(thickness_ratio=t,camber=r["camber"],camber_position=r["camber_position"],
            leading_edge_radius=0.5*t**2,trailing_edge_angle=12+20*t,
            aspect_ratio=r["aspect_ratio"],taper_ratio=r["taper_ratio"],
            sweep_angle=r["sweep_angle"],twist_angle=0.,dihedral_angle=3.)

PARAMETRIZATIONS = {"scalar": ScalarAirfoilParam, "naca": NACAFourDigitParam, "cst": CSTParam}
