from pydantic import BaseModel

class MoralValenceResponse(BaseModel):
    action_valence: float
    action_reasoning: str = ""
    action_factors: list[str] = []
    consequence_valence: float
    consequence_reasoning: str = ""
    consequence_factors: list[str] = []

class ModelInformation(BaseModel):
    model: str
    dataset: str
    task: str


class DatasetPrediction(BaseModel):
    id: int
    scenario: str
    action_valence: float
    consequence_valence: float