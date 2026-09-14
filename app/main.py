"""Mentor AI emotion classifier service.

Run: uv run uvicorn app.main:app --host 127.0.0.1 --port 8001
(port 8001 because the Mentor AI Express API already uses 8000)

MODEL_PATH selects the model: a .joblib file (TF-IDF pipeline) or a directory (DistilBERT).
Default: models/distilbert when trained and the `transformer` extra is installed, else models/emotion_classifier.joblib.
"""
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from src import config
from src.inference import load_predictor, predict_texts

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["predictor"] = load_predictor()
    yield
    state.clear()


app = FastAPI(title="Mentor AI Emotion Classifier", lifespan=lifespan)


class PredictionRequest(BaseModel):
    text: str = Field(..., max_length=config.MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value


class PredictionResponse(BaseModel):
    emotion: str
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    model_version: str


@app.get("/health")
def health() -> dict:
    predictor = state["predictor"]
    return {"status": "ok", "model_version": predictor.model_version, "classes": predictor.classes}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    predictor = state["predictor"]
    result = predict_texts(predictor, [req.text])[0]
    return PredictionResponse(
        emotion=result["emotion"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_version=predictor.model_version,
    )
