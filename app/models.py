from pydantic import BaseModel

class MoralValenceResponse(BaseModel):
    action_valence: float
    consequence_valence: float

class ModelInformation(BaseModel):
    model: str
    dataset: str
    task: str


class DatasetPrediction(BaseModel):
    id: int
    scenario: str
    action_valence: float
    consequence_valence: float