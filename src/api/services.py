import joblib
from pathlib import Path

from src.rag.rag_chain import generate_recommendation


MODEL_PATH = Path("models/churn_pipeline.joblib")


def predict_churn_and_recommend(customer_data):
    # Load trained ML pipeline
    model = joblib.load(MODEL_PATH)

    # Convert customer dictionary into DataFrame
    import pandas as pd

    customer_df = pd.DataFrame([customer_data])

    # ML prediction
    prediction = model.predict(customer_df)[0]

    # Churn probability
    probability = model.predict_proba(customer_df)[0][1]

    # Risk level
    if probability >= 0.70:
        risk_level = "HIGH"
    elif probability >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # RAG + Gemini recommendation
    recommendation = generate_recommendation(
        customer_data=customer_data,
        churn_probability=probability,
        risk_level=risk_level
    )

    return {
        "prediction": "CHURN" if prediction == 1 else "NO CHURN",
        "churn_probability": round(float(probability), 4),
        "churn_probability_percent": round(float(probability) * 100, 2),
        "risk_level": risk_level,
        "recommendation": recommendation
    }