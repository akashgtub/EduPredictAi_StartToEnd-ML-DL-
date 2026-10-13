
from typing import Any

import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load the trained pipeline
model = joblib.load("student_success_model.joblib")

# Read the feature names expected by the model
feature_columns = (
    model.named_steps["preprocessing"]
    .feature_names_in_
    .tolist()
)

app = FastAPI(
    title="EduPredict AI API",
    description="Student outcome prediction service",
    version="1.0.0"
)


class StudentData(BaseModel):
    features: dict[str, Any]


@app.get("/")
def home():
    return {
        "project": "EduPredict AI",
        "status": "running",
        "model": "Student Success Classifier"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict_student(data: StudentData):
    missing = [
        col for col in feature_columns
        if col not in data.features
    ]

    extra = [
        col for col in data.features
        if col not in feature_columns
    ]

    if missing or extra:
        raise HTTPException(
            status_code=400,
            detail={
                "missing_features": missing,
                "unexpected_features": extra
            }
        )

    try:
        student_df = pd.DataFrame(
            [data.features],
            columns=feature_columns
        )

        probabilities = model.predict_proba(student_df)[0]
        predicted_class = model.predict(student_df)[0]

        probability_map = {
            str(label): round(float(prob), 4)
            for label, prob in zip(
                model.classes_,
                probabilities
            )
        }

        return {
            "predicted_outcome": str(predicted_class),
            "class_probabilities": probability_map,
            "model_version": "1.0.0"
        }

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature values: {exc}"
        ) from exc
