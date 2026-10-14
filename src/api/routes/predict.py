# Implementation will be added step by step.
from fastapi import APIRouter
from src.api.services import predict_churn_and_recommend

router = APIRouter()


@router.post("/predict")
def predict_customer(customer_data: dict):
    result = predict_churn_and_recommend(customer_data)

    return result