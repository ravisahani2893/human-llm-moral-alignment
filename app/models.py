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


class CrossModelAxisResult(BaseModel):
    pearson_matrix: dict[str, dict[str, float | None]]
    spearman_matrix: dict[str, dict[str, float | None]]


class CrossModelPairwiseResult(BaseModel):
    model_a: str
    model_b: str
    action_pearson: float | None = None
    action_spearman: float | None = None
    consequence_pearson: float | None = None
    consequence_spearman: float | None = None


class CrossModelAgreementReport(BaseModel):
    """
    Cross-Model Agreement: model-vs-model correlation only. NOT to be
    confused with ModelAlignmentReport (Human-LLM Alignment: human-vs-model).
    """
    analysis: str
    prompt_version: str
    models: list[str]
    n_scenarios: int
    action: CrossModelAxisResult
    consequence: CrossModelAxisResult
    pairwise: list[CrossModelPairwiseResult]


class WilcoxonResult(BaseModel):
    statistic: float | None = None
    p_value: float | None = None
    mean_delta: float | None = None
    median_delta: float | None = None
    n: int


class VariantBiasResult(BaseModel):
    """
    One variant-pair comparison for one model (e.g. Male vs Female): a
    paired Wilcoxon signed-rank test on the model's own valence scores
    across the two variants of the same underlying scenario. Never reads
    or uses human annotations.
    """
    analysis: str
    model: str
    dataset: str
    variant_a: str
    variant_b: str
    n_scenarios: int
    action: WilcoxonResult
    consequence: WilcoxonResult


class VariantBiasReport(BaseModel):
    analysis: str
    model: str
    dataset: str
    variants: list[str]
    pairwise: list[VariantBiasResult]


class BiasEvalJobSnapshot(BaseModel):
    id: str
    model: str
    dataset: str
    prompt_version: str = "current"
    status: str
    error: str | None = None
    completed: int
    total: int
    created_at: float
    updated_at: float = 0.0
    csv_path: str


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