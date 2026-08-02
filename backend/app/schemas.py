from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("text must be at least 3 characters")
        if len(v) > 5000:
            raise ValueError("text must not exceed 5000 characters")
        return v


class PredictResponse(BaseModel):
    label: str
    confidence: float
