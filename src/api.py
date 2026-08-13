from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scripts.predict import predict
from enums import Checkpoint

app = FastAPI()


def resolve_model(name: str) -> str:
    """Match a requested model name to a Checkpoint member, ignoring case."""
    if name.upper() not in Checkpoint.__members__:
        raise HTTPException(
            status_code=400,
            detail=f"unknown model {name!r}; choose from {sorted(Checkpoint.__members__)}",
        )
    return name.upper()

class DetectClaimsRequest(BaseModel):
    model: str
    claims: list[str]


class DetectClaimsResponse(BaseModel):
    predictions: list


@app.post("/detect-claims")
async def detect_claims(request: DetectClaimsRequest):
    """
    Get predictions for claims using the specified model.
    
    Args:
        request: PredictionRequest containing model name and list of claims
        
    Returns:
        PredictionResponse with predictions
    """
    results = predict(resolve_model(request.model), request.claims)
    return DetectClaimsResponse(predictions=results)
