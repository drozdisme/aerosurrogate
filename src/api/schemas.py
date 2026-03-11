"""Pydantic schemas for AeroSurrogate v2.0 API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    thickness_ratio: Optional[float] = Field(0.12)
    camber: Optional[float] = Field(0.04)
    camber_position: Optional[float] = Field(0.4)
    leading_edge_radius: Optional[float] = Field(0.02)
    trailing_edge_angle: Optional[float] = Field(15.0)
    aspect_ratio: Optional[float] = Field(8.0)
    taper_ratio: Optional[float] = Field(0.5)
    sweep_angle: Optional[float] = Field(20.0)
    twist_angle: Optional[float] = Field(0.0)
    dihedral_angle: Optional[float] = Field(3.0)
    mach: float = Field(..., ge=0.0, le=5.0)
    reynolds: float = Field(..., ge=1e3, le=1e9)
    alpha: float = Field(..., ge=-30.0, le=30.0)
    beta: Optional[float] = Field(0.0)
    altitude: Optional[float] = Field(0.0)


class BatchPredictionInput(BaseModel):
    inputs: List[PredictionInput]


class TargetPrediction(BaseModel):
    value: float
    std: float


class ConfidenceInfo(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    level: str
    color: str


class PredictionOutput(BaseModel):
    predictions: Dict[str, TargetPrediction]
    confidence: ConfidenceInfo
    demo_mode: bool = False


class BatchPredictionOutput(BaseModel):
    results: List[PredictionOutput]
    count: int


class FieldPredictionOutput(BaseModel):
    x: List[float]
    Cp: List[float]
    n_points: int


class ModelMetrics(BaseModel):
    Cl_R2: float
    Cd_R2: float
    Cm_R2: float
    MAPE: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    demo_mode: bool = False
    deeponet_available: bool = False
    version: str = "2.0.0"
    model_version: Optional[str] = None
    metrics: Optional[ModelMetrics] = None


class CoeffSet(BaseModel):
    Cl: float
    Cd: float
    Cm: float
    K: float


class DeltaSet(BaseModel):
    Cl: float
    Cd: float
    Cm: float
    K: float


class CompareRequest(BaseModel):
    configA: PredictionInput
    configB: PredictionInput


class CompareResponse(BaseModel):
    A: CoeffSet
    B: CoeffSet
    delta: DeltaSet
    cpA: Optional[FieldPredictionOutput] = None
    cpB: Optional[FieldPredictionOutput] = None
    demo_mode: bool = False


class SweepPoint(BaseModel):
    alpha: float
    Cl: float
    Cd: float
    Cm: float
    K: float
    confidence: float


class SweepExportRequest(BaseModel):
    points: List[SweepPoint]
