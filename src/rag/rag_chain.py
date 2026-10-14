import os

from dotenv import load_dotenv
from google import genai

from src.rag.retriever import retrieve_documents


# Load environment variables
load_dotenv()


# Check API key
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file."
    )


# Create Gemini client
client = genai.Client(
    api_key=API_KEY
)


# Gemini model
MODEL_NAME = "gemini-3.6-flash"


def generate_recommendation(
    customer_data,
    churn_probability,
    risk_level
):
    """
    Generate an AI-powered customer retention recommendation
    using RAG + Gemini.
    """

    # --------------------------------------------------
    # 1. Create search query
    # --------------------------------------------------

    query = f"""
    Customer churn risk analysis.

    Churn probability: {churn_probability:.2%}
    Risk level: {risk_level}

    Customer information:
    {customer_data}

    Find relevant retention policies, product information,
    support policies, and retention recommendations.
    """

    # --------------------------------------------------
    # 2. Retrieve relevant documents from ChromaDB
    # --------------------------------------------------

    documents = retrieve_documents(
        query,
        k=4
    )

    # --------------------------------------------------
    # 3. Build RAG context
    # --------------------------------------------------

    context_parts = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        content = document.page_content

        context_parts.append(
            f"Source: {source}\n"
            f"Content:\n{content}"
        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------
    # 4. Create Gemini prompt
    # --------------------------------------------------

    prompt = f"""
You are an AI Customer Retention Analyst.

Analyze the customer's churn risk and provide practical
retention recommendations.

CUSTOMER DATA
--------------
{customer_data}

CHURN PROBABILITY
-----------------
{churn_probability:.2%}

RISK LEVEL
----------
{risk_level}

RETRIEVED KNOWLEDGE BASE
------------------------
{context}

INSTRUCTIONS
------------

1. Explain why the customer is considered at risk.

2. Identify the most important risk factors from the
   customer data.

3. Recommend 2-4 practical retention actions.

4. Base recommendations primarily on the retrieved
   knowledge base.

5. Do not invent company policies.

6. Do not automatically recommend discounts.

7. Recommendations should be realistic and
   customer-specific.

8. Clearly distinguish between model predictions and
   business recommendations.

9. Keep the response concise and professional.

Return the response using this format:

Risk Summary:
<short explanation>

Key Risk Factors:
- factor 1
- factor 2
- factor 3

Recommended Actions:
1. action
2. action
3. action

Reasoning:
<short explanation of why these actions are appropriate>
"""

    # --------------------------------------------------
    # 5. Generate response using Gemini
    # --------------------------------------------------

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# ------------------------------------------------------
# Test the RAG + Gemini pipeline
# ------------------------------------------------------

if __name__ == "__main__":

    sample_customer = {
        "Tenure Months": 5,
        "Monthly Charges": 90.0,
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Tech Support": "No",
        "Contract": "Month-to-month",
        "Payment Method": "Electronic check"
    }

    sample_probability = 0.7484

    sample_risk = "HIGH"

    result = generate_recommendation(
        customer_data=sample_customer,
        churn_probability=sample_probability,
        risk_level=sample_risk
    )

    print("\n")
    print("=" * 60)
    print("AI RETENTION RECOMMENDATION")
    print("=" * 60)

    print(result)