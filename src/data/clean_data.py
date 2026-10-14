# Implementation will be added step by step.
import pandas as pd
from pathlib import Path


# Project paths
RAW_FILE = Path("data/raw/customer_churn.csv")
PROCESSED_FILE = Path("data/processed/customer_churn_clean.csv")


def load_data():
    """Load the raw customer churn dataset."""
    df = pd.read_csv(RAW_FILE)
    print(f"Loaded dataset: {df.shape}")
    return df


def clean_data(df):
    """Clean the customer churn dataset."""

    # 1. Convert Total Charges from text to numeric
    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors="coerce"
    )

    # 2. Fill missing Total Charges
    # Customers with 0 tenure have no accumulated charges.
    df["Total Charges"] = df["Total Charges"].fillna(0)

    # 3. Remove irrelevant / constant columns
    columns_to_drop = [
        "Count",
        "Country",
        "State",
        "CustomerID",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Zip Code",
        "City",
    ]

    df = df.drop(columns=columns_to_drop)

    # 4. Remove target leakage columns
    leakage_columns = [
        "Churn Value",
        "Churn Score",
        "Churn Reason",
    ]

    df = df.drop(columns=leakage_columns)

    # 5. Convert target from Yes/No to 1/0
    df["Churn Label"] = df["Churn Label"].map({
        "Yes": 1,
        "No": 0
    })

    return df


def save_data(df):
    """Save cleaned dataset."""
    PROCESSED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(PROCESSED_FILE, index=False)

    print(f"Cleaned dataset saved to: {PROCESSED_FILE}")
    print(f"Final shape: {df.shape}")


def main():
    df = load_data()

    print("\nBefore cleaning:")
    print(df.shape)

    df = clean_data(df)

    print("\nAfter cleaning:")
    print(df.shape)

    print("\nRemaining missing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\nTarget distribution:")
    print(df["Churn Label"].value_counts())

    save_data(df)


if __name__ == "__main__":
    main()