import { useState } from "react";
import axios from "axios";
import "./App.css";

const initialData = {
  Gender: "Male",
  "Senior Citizen": "No",
  Partner: "No",
  Dependents: "No",
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
  Contract: "Month-to-month",
  "Paperless Billing": "Yes",
  "Payment Method": "Electronic check",
  "Monthly Charges": 90,
  "Total Charges": 450,
  CLTV: 3500,
};

const options = {
  Gender: ["Male", "Female"],
  "Senior Citizen": ["No", "Yes"],
  Partner: ["No", "Yes"],
  Dependents: ["No", "Yes"],
  "Phone Service": ["No", "Yes"],
  "Multiple Lines": ["No phone service", "No", "Yes"],
  "Internet Service": ["DSL", "Fiber optic", "No"],
  "Online Security": ["No internet service", "No", "Yes"],
  "Online Backup": ["No internet service", "No", "Yes"],
  "Device Protection": ["No internet service", "No", "Yes"],
  "Tech Support": ["No internet service", "No", "Yes"],
  "Streaming TV": ["No internet service", "No", "Yes"],
  "Streaming Movies": ["No internet service", "No", "Yes"],
  Contract: ["Month-to-month", "One year", "Two year"],
  "Paperless Billing": ["No", "Yes"],
  "Payment Method": [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
  ],
};

function parseRecommendation(text) {
  const sections = {
    riskSummary: "",
    riskFactors: [],
    actions: [],
    reasoning: "",
  };

  if (!text) {
    return sections;
  }

  const riskSummaryMatch = text.match(
    /Risk Summary:\s*([\s\S]*?)(?=Key Risk Factors:|Recommended Actions:|Reasoning:|$)/i
  );

  const riskFactorsMatch = text.match(
    /Key Risk Factors:\s*([\s\S]*?)(?=Recommended Actions:|Reasoning:|$)/i
  );

  const actionsMatch = text.match(
    /Recommended Actions:\s*([\s\S]*?)(?=Reasoning:|$)/i
  );

  const reasoningMatch = text.match(
    /Reasoning:\s*([\s\S]*)/i
  );

  if (riskSummaryMatch) {
    sections.riskSummary = riskSummaryMatch[1].trim();
  }

  if (riskFactorsMatch) {
    sections.riskFactors = riskFactorsMatch[1]
      .split("\n")
      .map((line) => line.replace(/^[-•]\s*/, "").trim())
      .filter(Boolean);
  }

  if (actionsMatch) {
    sections.actions = actionsMatch[1]
      .split("\n")
      .map((line) => line.replace(/^\d+[.)]\s*/, "").trim())
      .filter(Boolean);
  }

  if (reasoningMatch) {
    sections.reasoning = reasoningMatch[1].trim();
  }

  return sections;
}

function App() {
  const [formData, setFormData] = useState(initialData);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData((previous) => ({
      ...previous,
      [name]: value,
    }));
  };

  const predictChurn = async () => {
    try {
      setLoading(true);
      setResult(null);

      const customerData = {
        ...formData,
        "Tenure Months": Number(formData["Tenure Months"]),
        "Monthly Charges": Number(formData["Monthly Charges"]),
        "Total Charges": Number(formData["Total Charges"]),
        CLTV: Number(formData.CLTV),
      };

      const response = await axios.post(
        "http://127.0.0.1:8000/api/predict",
        customerData
      );

      setResult(response.data);
    } catch (error) {
      console.error(error);

      alert(
        "Unable to connect to the FastAPI backend. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const recommendation = result
    ? parseRecommendation(result.recommendation)
    : null;

  const riskClass = result
    ? result.risk_level.toLowerCase()
    : "";

  return (
    <div className="app">

      {/* HEADER */}
      <header>
        <div className="header-content">
          <h1>AI Customer Churn Analytics</h1>
          <p>
            Machine Learning + RAG-powered Customer Retention System
          </p>
        </div>
      </header>

      {/* MAIN */}
      <main>

        {/* CUSTOMER FORM */}
        <section className="card">

          <div className="section-title">
            <h2>Customer Information</h2>
            <span>Customer Profile</span>
          </div>

          <div className="form-grid">

            {Object.keys(formData).map((key) => (
              <div className="form-group" key={key}>

                <label>{key}</label>

                {options[key] ? (
                  <select
                    name={key}
                    value={formData[key]}
                    onChange={handleChange}
                  >
                    {options[key].map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    name={key}
                    value={formData[key]}
                    onChange={handleChange}
                  />
                )}

              </div>
            ))}

          </div>

          <button
            className="predict-button"
            onClick={predictChurn}
            disabled={loading}
          >
            {loading ? "Analyzing Customer..." : "Predict Churn"}
          </button>

        </section>


        {/* RESULTS */}
        {result && (
          <section className="card result-card">

            <div className="section-title">
              <h2>Prediction Result</h2>
              <span>AI Analysis</span>
            </div>


            {/* RISK */}
            <div className={`risk-card ${riskClass}`}>

              <span className="risk-label">
                CHURN RISK
              </span>

              <strong>
                {result.risk_level}
              </strong>

            </div>


            {/* METRICS */}
            <div className="metrics">

              <div className="metric-card">

                <span>Prediction</span>

                <strong>
                  {result.prediction}
                </strong>

              </div>

              <div className="metric-card">

                <span>Churn Probability</span>

                <strong>
                  {result.churn_probability_percent}%
                </strong>

              </div>

            </div>


            {/* PROBABILITY BAR */}
            <div className="probability-section">

              <div className="probability-header">

                <span>Churn Probability</span>

                <strong>
                  {result.churn_probability_percent}%
                </strong>

              </div>

              <div className="probability-bar">

                <div
                  className={`probability-fill ${riskClass}`}
                  style={{
                    width: `${result.churn_probability_percent}%`,
                  }}
                />

              </div>

            </div>


            {/* AI RECOMMENDATION */}
            <div className="ai-section">

              <h2>
                AI Retention Recommendation
              </h2>


              {/* RISK SUMMARY */}
              {recommendation.riskSummary && (
                <div className="ai-block">

                  <h3>Risk Summary</h3>

                  <p>
                    {recommendation.riskSummary}
                  </p>

                </div>
              )}


              {/* RISK FACTORS */}
              {recommendation.riskFactors.length > 0 && (
                <div className="ai-block">

                  <h3>Key Risk Factors</h3>

                  <ul>
                    {recommendation.riskFactors.map(
                      (factor, index) => (
                        <li key={index}>
                          {factor}
                        </li>
                      )
                    )}
                  </ul>

                </div>
              )}


              {/* ACTIONS */}
              {recommendation.actions.length > 0 && (
                <div className="ai-block">

                  <h3>Recommended Actions</h3>

                  <ol>
                    {recommendation.actions.map(
                      (action, index) => (
                        <li key={index}>
                          {action}
                        </li>
                      )
                    )}
                  </ol>

                </div>
              )}


              {/* REASONING */}
              {recommendation.reasoning && (
                <div className="ai-block">

                  <h3>Reasoning</h3>

                  <p>
                    {recommendation.reasoning}
                  </p>

                </div>
              )}

            </div>

          </section>
        )}

      </main>

    </div>
  );
}

export default App;