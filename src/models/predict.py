# Implementation will be added step by step.
import pandas as pd
import joblib

from pathlib import Path


MODEL_FILE = Path("models/churn_pipeline.joblib")


def load_model():
    """Load the trained churn prediction pipeline."""

    model = joblib.load(MODEL_FILE)

    print("Churn model loaded successfully.")

    return model


def predict_churn(customer_data):
    """
    Predict churn for a single customer.

    Parameters
    ----------
    customer_data : dict
        Customer information.

    Returns
    -------
    dict
        Churn prediction, probability and risk level.
    """

    model = load_model()

    # Convert dictionary to DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Get prediction
    prediction = model.predict(customer_df)[0]

    # Get churn probability
    probability = model.predict_proba(
        customer_df
    )[0][1]

    # Determine risk level
    if probability >= 0.70:
        risk_level = "HIGH"

    elif probability >= 0.40:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    result = {
        "churn_prediction": int(prediction),
        "churn_probability": round(
            float(probability),
            4
        ),
        "churn_percentage": round(
            float(probability * 100),
            2
        ),
        "risk_level": risk_level
    }

    return result


if __name__ == "__main__":

    sample_customer = {
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": 5,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 90.0,
        "Total Charges": 450.0,
        "CLTV": 3500
    }

    result = predict_churn(
        sample_customer
    )

    print("\n" + "=" * 50)
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 50)

    print(
        f"Prediction: "
        f"{'CHURN' if result['churn_prediction'] == 1 else 'NO CHURN'}"
    )

    print(
        f"Churn Probability: "
        f"{result['churn_percentage']}%"
    )

    print(
        f"Risk Level: "
        f"{result['risk_level']}"
    )