# Implementation will be added step by step.
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report
)


DATA_FILE = Path("data/processed/customer_churn_clean.csv")
MODEL_FILE = Path("models/churn_pipeline.joblib")

OUTPUT_DIR = Path("outputs/figures")


def load_data():
    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=["Churn Label"])
    y = df["Churn Label"]

    return X, y


def load_model():
    model = joblib.load(MODEL_FILE)

    print("Model loaded successfully.")

    return model


def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def confusion_matrix_plot(model, X_test, y_test):

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Customer Churn Confusion Matrix")

    plt.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print(
        "Saved: outputs/figures/confusion_matrix.png"
    )


def roc_curve_plot(model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"Gradient Boosting (AUC = {roc_auc:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "roc_curve.png",
        dpi=300
    )

    plt.close()

    print(
        "Saved: outputs/figures/roc_curve.png"
    )


def classification_report_output(
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "No Churn",
            "Churn"
        ]
    )

    print("\nClassification Report:")
    print(report)

    report_file = Path(
        "outputs/reports/classification_report.txt"
    )

    report_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        report_file,
        "w"
    ) as file:

        file.write(report)

    print(
        f"Saved: {report_file}"
    )
def feature_importance(model):

    # Get the preprocessing step
    preprocessor = model.named_steps["preprocessor"]

    # Get the trained model
    classifier = model.named_steps["model"]

    # Get feature names after OneHotEncoding
    feature_names = preprocessor.get_feature_names_out()

    # Get feature importance
    importances = classifier.feature_importances_

    # Create dataframe
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    # Sort by importance
    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 20 Important Features:")
    print(
        importance_df.head(20).to_string(index=False)
    )

    # Save CSV
    report_path = Path(
        "outputs/reports/feature_importance.csv"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    importance_df.to_csv(
        report_path,
        index=False
    )

    # Plot top 20
    top_features = importance_df.head(20).sort_values(
        by="Importance"
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 20 Customer Churn Feature Importance")

    plt.tight_layout()

    figure_path = Path(
        "outputs/figures/feature_importance.png"
    )

    plt.savefig(
        figure_path,
        dpi=300
    )

    plt.close()

    print(f"\nSaved: {report_path}")
    print(f"Saved: {figure_path}")

def main():

    print("=" * 60)
    print("CUSTOMER CHURN MODEL EVALUATION")
    print("=" * 60)

    X, y = load_data()

    model = load_model()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    confusion_matrix_plot(
        model,
        X_test,
        y_test
    )

    roc_curve_plot(
        model,
        X_test,
        y_test
    )

    classification_report_output(
        model,
        X_test,
        y_test
    )

    feature_importance(model)

    

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()