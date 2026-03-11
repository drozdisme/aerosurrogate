"""Telemetry ingestion with ISA atmosphere model for AeroSurrogate v3.0."""
import logging, math, time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_T0=288.15; _P0=101325.0; _R=287.058; _L=0.0065
_MU0=1.716e-5; _T_MU=273.15; _C=110.4

def isa_temperature(h): return _T0 - _L * max(0., min(h, 11000.))
def isa_pressure(h): T=isa_temperature(h); return _P0*(T/_T0)**(9.80665/(_L*_R))
def isa_density(h): T=isa_temperature(h); return isa_pressure(h)/(_R*T)
def sutherland_viscosity(T): return _MU0*((T/_T_MU)**1.5)*((_T_MU+_C)/(T+_C))
def compute_reynolds(mach, altitude_m=0., chord_m=1.):
    T=isa_temperature(altitude_m); rho=isa_density(altitude_m)
    a=math.sqrt(1.4*_R*T); mu=sutherland_viscosity(T)
    return rho*mach*a*chord_m/mu

@dataclass
class TelemetryRecord:
    timestamp: float; mach: float; alpha: float
    altitude_m: float=0.; beta: float=0.; reynolds: float=1e6
    thickness_ratio: float=0.12; camber: float=0.04; camber_position: float=0.40
    leading_edge_radius: float=0.02; trailing_edge_angle: float=15.0
    aspect_ratio: float=8.0; taper_ratio: float=0.5; sweep_angle: float=20.0
    twist_angle: float=0.; dihedral_angle: float=3.; source: str="telemetry"

    def to_dict(self):
        return {k: getattr(self, k) for k in
                ["mach","reynolds","alpha","beta","altitude_m",
                 "thickness_ratio","camber","camber_position",
                 "leading_edge_radius","trailing_edge_angle",
                 "aspect_ratio","taper_ratio","sweep_angle",
                 "twist_angle","dihedral_angle"]}

class TelemetryIngestion:
    GEOM_DEFAULTS = dict(thickness_ratio=0.12, camber=0.04, camber_position=0.40,
        leading_edge_radius=0.02, trailing_edge_angle=15.0,
        aspect_ratio=8.0, taper_ratio=0.5, sweep_angle=20.0,
        twist_angle=0., dihedral_angle=3.)

    def __init__(self, chord_m=1.0):
        self.chord_m = chord_m

    def ingest(self, raw: Dict) -> TelemetryRecord:
        r = dict(raw)
        # normalise alpha aliases
        alpha = float(r.get("alpha", r.get("alpha_deg", r.get("aoa", 0.))))
        mach  = float(r.get("mach", r.get("mach_number", r.get("M", 0.))))
        alt   = float(r.get("altitude_m", r.get("altitude", r.get("alt", 0.))))
        beta  = float(r.get("beta", r.get("beta_deg", 0.)))
        ts    = float(r.get("timestamp", time.time()))
        re    = float(r["reynolds"]) if r.get("reynolds") else compute_reynolds(mach, alt, self.chord_m)
        geom  = {k: float(r.get(k, v)) for k, v in self.GEOM_DEFAULTS.items()}
        return TelemetryRecord(timestamp=ts, mach=mach, alpha=alpha,
            altitude_m=alt, beta=beta, reynolds=re, **geom)

    def ingest_batch(self, records):
        return [self.ingest(r) for r in records]
