# Implementation will be added step by step.
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path("data/processed/customer_churn_clean.csv")
OUTPUT_DIR = Path("data/processed")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load cleaned customer churn data."""

    df = pd.read_csv(DATA_FILE)

    print(f"Dataset shape: {df.shape}")

    return df


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

def prepare_data(df):
    """Separate input features and target variable."""

    X = df.drop(columns=["Churn Label"])

    y = df["Churn Label"]

    print("\nFeatures:")
    print(X.columns.tolist())

    print("\nTarget:")
    print("Churn Label")

    return X, y


# ============================================================
# IDENTIFY COLUMN TYPES
# ============================================================

def identify_columns(X):

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("\nNumerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)

    return numerical_features, categorical_features


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

def create_preprocessor(
    numerical_features,
    categorical_features
):

    numerical_transformer = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
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
                "num",
                numerical_transformer,
                numerical_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocessor


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain/Test Split:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load dataset
    df = load_data()

    # Prepare X and y
    X, y = prepare_data(df)

    # Identify columns
    numerical_features, categorical_features = (
        identify_columns(X)
    )

    # Create preprocessing pipeline
    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features
    )

    # Train/Test split
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # Fit preprocessing only on training data
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # Transform test data
    X_test_processed = preprocessor.transform(
        X_test
    )

    print("\nProcessed feature shapes:")
    print(
        f"X_train: {X_train_processed.shape}"
    )
    print(
        f"X_test: {X_test_processed.shape}"
    )

    print("\nFeature engineering completed successfully.")


if __name__ == "__main__":
    main()