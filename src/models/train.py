# Implementation will be added step by step.
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path("data/processed/customer_churn_clean.csv")
MODEL_DIR = Path("models")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATA_FILE)

    print(f"Dataset shape: {df.shape}")

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    X = df.drop(columns=["Churn Label"])

    y = df["Churn Label"]

    return X, y


# ============================================================
# IDENTIFY FEATURES
# ============================================================

def identify_features(X):

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    print("\nNumerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)

    return numerical_features, categorical_features


# ============================================================
# CREATE PREPROCESSOR
# ============================================================

def create_preprocessor(
    numerical_features,
    categorical_features
):

    numerical_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# DEFINE MODELS
# ============================================================

def get_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }

    return models


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    name,
    pipeline,
    X_test,
    y_test
):

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No Churn",
                "Churn"
            ],
            zero_division=0
        )
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc
    }


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):

    models = get_models()

    results = []

    trained_pipelines = {}

    for name, model in models.items():

        print("\n" + "-" * 60)
        print(f"Training {name}...")
        print("-" * 60)

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        result = evaluate_model(
            name,
            pipeline,
            X_test,
            y_test
        )

        results.append(result)

        trained_pipelines[name] = pipeline

    return results, trained_pipelines


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_best_model(
    results,
    trained_pipelines
):

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="ROC-AUC",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    best_model_name = results_df.iloc[0]["Model"]

    best_pipeline = trained_pipelines[
        best_model_name
    ]

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = MODEL_DIR / "churn_pipeline.joblib"

    joblib.dump(
        best_pipeline,
        model_path
    )

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(f"Model: {best_model_name}")
    print(
        f"ROC-AUC: "
        f"{results_df.iloc[0]['ROC-AUC']:.4f}"
    )

    print(
        f"\nModel saved to: {model_path}"
    )

    # Save comparison results
    results_df.to_csv(
        "outputs/reports/model_comparison.csv",
        index=False
    )

    return best_model_name


# ============================================================
# MAIN
# ============================================================

def main():

    # Create directories
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    Path("outputs/reports").mkdir(
        parents=True,
        exist_ok=True
    )

    # Load data
    df = load_data()

    # Prepare X and y
    X, y = prepare_data(df)

    # Identify feature types
    numerical_features, categorical_features = (
        identify_features(X)
    )

    # Create preprocessing
    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features
    )

    # Train/test split
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print("\nTrain/Test Split:")
    print(
        f"Training samples: {len(X_train)}"
    )
    print(
        f"Testing samples: {len(X_test)}"
    )

    # Train models
    results, trained_pipelines = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    # Save best model
    save_best_model(
        results,
        trained_pipelines
    )


if __name__ == "__main__":
    main()