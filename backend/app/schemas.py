from pydantic import BaseModel

#defines JSON structure to expect
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float