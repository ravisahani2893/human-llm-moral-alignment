from pydantic import BaseModel

class MoralValenceResponse(BaseModel):
    action_valence: float
    action_reasoning: str = ""
    action_factors: list[str] = []
    consequence_valence: float
    consequence_reasoning: str = ""
    consequence_factors: list[str] = []

class AlignmentAxisResult(BaseModel):
    ccc: float | None = None
    mae: float | None = None
    rmse: float | None = None
    pearson: float | None = None
    spearman: float | None = None


class ModelAlignmentReport(BaseModel):
    model: str
    prompt_version: str
    n_scenarios: int
    action_results: AlignmentAxisResult
    consequence_results: AlignmentAxisResult


class ModelInformation(BaseModel):
    model: str
    dataset: str
    task: str


class DatasetPrediction(BaseModel):
    id: int
    scenario: str
    action_valence: float
    consequence_valence: float


class JobSnapshot(BaseModel):
    id: str
    status: str
    error: str | None = None
    models: list[str]
    total: int
    completed_per_model: dict[str, int]
    errors_per_model: dict[str, int]
    created_at: float
    updated_at: float = 0.0
    csv_path: str


class ExportJobSnapshot(BaseModel):
    id: str
    model: str
    prompt_version: str = "current"
    status: str
    error: str | None = None
    completed: int
    total: int
    created_at: float
    updated_at: float = 0.0
    csv_path: str


class AxisMetrics(BaseModel):
    n: int
    pearson_r: float | None = None
    spearman_r: float | None = None
    ccc: float | None = None
    mae: float | None = None
    rmse: float | None = None
    sign_agreement: float | None = None
    mean_bias: float | None = None


class ModelMetrics(BaseModel):
    action: AxisMetrics
    consequence: AxisMetrics
    combined: AxisMetrics


class StratumMetrics(BaseModel):
    value: str
    n: int
    action_mae: float | None = None
    action_sign_agreement: float | None = None
    consequence_mae: float | None = None
    consequence_sign_agreement: float | None = None


class AlignmentReport(BaseModel):
    n_scenarios: int
    models: dict[str, ModelMetrics]
    cross_model_agreement: dict[str, dict[str, float | None]]
    stratified: dict[str, dict[str, list[StratumMetrics]]]