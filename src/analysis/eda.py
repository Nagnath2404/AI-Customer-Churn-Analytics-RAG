import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path("data/processed/customer_churn_clean.csv")
OUTPUT_DIR = Path("outputs/figures")


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the cleaned customer churn dataset."""
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset loaded successfully: {df.shape}")
    return df


# ============================================================
# 1. CHURN DISTRIBUTION
# ============================================================

def analyze_churn_distribution(df):
    """Analyze overall customer churn distribution."""

    counts = df["Churn Label"].value_counts().sort_index()

    percentages = (
        df["Churn Label"]
        .value_counts(normalize=True)
        .sort_index() * 100
    )

    print("\n" + "=" * 60)
    print("1. CHURN DISTRIBUTION")
    print("=" * 60)

    print(f"Non-churned customers: {counts.get(0, 0)} "
          f"({percentages.get(0, 0):.2f}%)")

    print(f"Churned customers: {counts.get(1, 0)} "
          f"({percentages.get(1, 0):.2f}%)")

    plt.figure(figsize=(7, 5))

    counts.plot(kind="bar")

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.xticks(
        [0, 1],
        ["No Churn", "Churn"],
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "churn_distribution.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 2. CHURN BY CONTRACT
# ============================================================

def churn_by_contract(df):
    """Calculate and plot churn rate by contract."""

    churn_rate = (
        df.groupby("Contract")["Churn Label"]
        .mean() * 100
    )

    print("\n" + "=" * 60)
    print("2. CHURN RATE BY CONTRACT")
    print("=" * 60)

    print(churn_rate.round(2))

    plt.figure(figsize=(8, 5))

    churn_rate.plot(kind="bar")

    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "churn_by_contract.png",
        dpi=300
    )

    plt.show()

    return churn_rate


# ============================================================
# 3. CHURN BY TENURE
# ============================================================

def churn_by_tenure(df):
    """Calculate churn rate by customer tenure."""

    bins = [-1, 12, 24, 48, 72]

    labels = [
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49-72 months"
    ]

    df["Tenure Group"] = pd.cut(
        df["Tenure Months"],
        bins=bins,
        labels=labels
    )

    churn_rate = (
        df.groupby(
            "Tenure Group",
            observed=True
        )["Churn Label"]
        .mean() * 100
    )

    print("\n" + "=" * 60)
    print("3. CHURN RATE BY TENURE")
    print("=" * 60)

    print(churn_rate.round(2))

    plt.figure(figsize=(8, 5))

    churn_rate.plot(kind="bar")

    plt.title("Churn Rate by Customer Tenure")
    plt.xlabel("Tenure Group")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "churn_by_tenure.png",
        dpi=300
    )

    plt.show()

    return churn_rate


# ============================================================
# 4. CHURN BY INTERNET SERVICE
# ============================================================

def churn_by_internet_service(df):
    """Calculate churn rate by internet service."""

    churn_rate = (
        df.groupby("Internet Service")["Churn Label"]
        .mean() * 100
    )

    print("\n" + "=" * 60)
    print("4. CHURN RATE BY INTERNET SERVICE")
    print("=" * 60)

    print(churn_rate.round(2))

    plt.figure(figsize=(8, 5))

    churn_rate.plot(kind="bar")

    plt.title("Churn Rate by Internet Service")
    plt.xlabel("Internet Service")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "churn_by_internet_service.png",
        dpi=300
    )

    plt.show()

    return churn_rate


# ============================================================
# 5. CHURN BY PAYMENT METHOD
# ============================================================

def churn_by_payment_method(df):
    """Calculate churn rate by payment method."""

    churn_rate = (
        df.groupby("Payment Method")["Churn Label"]
        .mean() * 100
    ).sort_values(ascending=False)

    print("\n" + "=" * 60)
    print("5. CHURN RATE BY PAYMENT METHOD")
    print("=" * 60)

    print(churn_rate.round(2))

    plt.figure(figsize=(10, 6))

    churn_rate.plot(kind="bar")

    plt.title("Churn Rate by Payment Method")
    plt.xlabel("Payment Method")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "churn_by_payment_method.png",
        dpi=300
    )

    plt.show()

    return churn_rate


# ============================================================
# 6. CHURN BY GENDER
# ============================================================

def churn_by_gender(df):
    """Calculate churn rate by gender."""

    churn_rate = (
        df.groupby("Gender")["Churn Label"]
        .mean() * 100
    )

    print("\n" + "=" * 60)
    print("6. CHURN RATE BY GENDER")
    print("=" * 60)

    print(churn_rate.round(2))

    plt.figure(figsize=(7, 5))

    churn_rate.plot(kind="bar")

    plt.title("Churn Rate by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "churn_by_gender.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 7. CHURN BY SENIOR CITIZEN
# ============================================================

def churn_by_senior_citizen(df):
    """Calculate churn rate by senior citizen status."""

    churn_rate = (
        df.groupby("Senior Citizen")["Churn Label"]
        .mean() * 100
    )

    print("\n" + "=" * 60)
    print("7. CHURN RATE BY SENIOR CITIZEN")
    print("=" * 60)

    print(churn_rate.round(2))

    plt.figure(figsize=(7, 5))

    churn_rate.plot(kind="bar")

    plt.title("Churn Rate by Senior Citizen Status")
    plt.xlabel("Senior Citizen")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(
        [0, 1],
        ["Non-Senior", "Senior"],
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "churn_by_senior_citizen.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 8. CHURN BY PARTNER
# ============================================================

def churn_by_partner(df):
    """Calculate churn rate by partner status."""

    churn_rate = (
        df.groupby("Partner")["Churn Label"]
        .mean() * 100
    )

    print("\n" + "=" * 60)
    print("8. CHURN RATE BY PARTNER STATUS")
    print("=" * 60)

    print(churn_rate.round(2))

    plt.figure(figsize=(7, 5))

    churn_rate.plot(kind="bar")

    plt.title("Churn Rate by Partner Status")
    plt.xlabel("Partner")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "churn_by_partner.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 9. CHURN BY DEPENDENTS
# ============================================================

def churn_by_dependents(df):
    """Calculate churn rate by dependents."""

    churn_rate = (
        df.groupby("Dependents")["Churn Label"]
        .mean() * 100
    )

    print("\n" + "=" * 60)
    print("9. CHURN RATE BY DEPENDENTS")
    print("=" * 60)

    print(churn_rate.round(2))

    plt.figure(figsize=(7, 5))

    churn_rate.plot(kind="bar")

    plt.title("Churn Rate by Dependents")
    plt.xlabel("Dependents")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "churn_by_dependents.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 10. MONTHLY CHARGES VS CHURN
# ============================================================

def monthly_charges_by_churn(df):
    """Compare monthly charges between churned and non-churned customers."""

    print("\n" + "=" * 60)
    print("10. MONTHLY CHARGES BY CHURN")
    print("=" * 60)

    result = (
        df.groupby("Churn Label")["Monthly Charges"]
        .agg(["mean", "median"])
    )

    print(result.round(2))

    plt.figure(figsize=(7, 5))

    df.boxplot(
        column="Monthly Charges",
        by="Churn Label"
    )

    plt.title("Monthly Charges vs Churn")
    plt.suptitle("")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "monthly_charges_vs_churn.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 11. TENURE VS CHURN
# ============================================================

def tenure_by_churn(df):
    """Compare customer tenure between churn groups."""

    print("\n" + "=" * 60)
    print("11. TENURE BY CHURN")
    print("=" * 60)

    result = (
        df.groupby("Churn Label")["Tenure Months"]
        .agg(["mean", "median"])
    )

    print(result.round(2))

    plt.figure(figsize=(7, 5))

    df.boxplot(
        column="Tenure Months",
        by="Churn Label"
    )

    plt.title("Customer Tenure vs Churn")
    plt.suptitle("")
    plt.xlabel("Churn")
    plt.ylabel("Tenure Months")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "tenure_vs_churn.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 12. CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(df):
    """Analyze correlations between numerical variables."""

    numeric_columns = [
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "CLTV",
        "Churn Label"
    ]

    correlation = df[numeric_columns].corr()

    print("\n" + "=" * 60)
    print("12. CORRELATION MATRIX")
    print("=" * 60)

    print(correlation.round(2))

    plt.figure(figsize=(8, 6))

    plt.imshow(correlation, interpolation="nearest")

    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title("Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlation_matrix.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 13. NUMERICAL SUMMARY BY CHURN
# ============================================================

def numerical_summary(df):
    """Display numerical statistics by churn."""

    numerical_columns = [
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "CLTV"
    ]

    print("\n" + "=" * 60)
    print("13. NUMERICAL SUMMARY BY CHURN")
    print("=" * 60)

    summary = (
        df.groupby("Churn Label")[numerical_columns]
        .mean()
    )

    print(summary.round(2))


# ============================================================
# MAIN
# ============================================================

def main():

    # Create output directory
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load data
    df = load_data()

    print("\nDataset columns:")
    print(df.columns.tolist())

    print("\nDataset information:")
    print(df.info())

    # Run EDA analyses
    analyze_churn_distribution(df)

    churn_by_contract(df)

    churn_by_tenure(df)

    churn_by_internet_service(df)

    churn_by_payment_method(df)

    churn_by_gender(df)

    churn_by_senior_citizen(df)

    churn_by_partner(df)

    churn_by_dependents(df)

    monthly_charges_by_churn(df)

    tenure_by_churn(df)

    correlation_analysis(df)

    numerical_summary(df)

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nGraphs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()