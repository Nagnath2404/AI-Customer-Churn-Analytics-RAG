from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.predict import router as predict_router


app = FastAPI(
    title="AI Customer Churn Analytics + RAG",
    description="ML churn prediction with RAG-powered retention recommendations",
    version="1.0.0"
)


# Allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "AI Customer Churn Analytics + RAG API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


app.include_router(
    predict_router,
    prefix="/api"
)