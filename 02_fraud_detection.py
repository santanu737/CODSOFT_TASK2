# ============================================================
# CodSoft ML Internship - Task 2
# Credit Card Fraud Detection
# Author: Santanu Mondal
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42
DATASET_FILE = "creditcard.csv"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION")
print("=" * 60)

print("\nCurrent working directory:")
print(os.getcwd())

print("\nLoading dataset...")

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        f"\n'{DATASET_FILE}' was not found.\n"
        f"Place creditcard.csv inside:\n{os.getcwd()}"
    )

df = pd.read_csv(DATASET_FILE)

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 3. DATASET INFORMATION
# ============================================================

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nClass Distribution:")
print(df["Class"].value_counts())


# ============================================================
# 4. CHECK CLASS IMBALANCE
# ============================================================

fraud_percent = df["Class"].mean() * 100

print(
    f"\nFraud transactions are only "
    f"{fraud_percent:.3f}% of the dataset."
)

print("This is a highly imbalanced classification problem.")


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\nCreating EDA plots...")

plt.figure(figsize=(12, 5))


# Class distribution
plt.subplot(1, 2, 1)

sns.countplot(
    x="Class",
    data=df
)

plt.yscale("log")

plt.title("Legitimate vs Fraud Transactions")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Count (Log Scale)")


# Amount distribution
plt.subplot(1, 2, 2)

sns.boxplot(
    x="Class",
    y="Amount",
    data=df
)

plt.ylim(0, 300)

plt.title("Transaction Amount by Class")
plt.xlabel("Class")
plt.ylabel("Transaction Amount")


plt.tight_layout()

plt.savefig(
    "eda_overview.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print("Saved: eda_overview.png")


# ============================================================
# 6. SCALE TIME AND AMOUNT
# ============================================================

print("\nScaling Time and Amount...")

scaler = StandardScaler()

df["scaled_amount"] = scaler.fit_transform(
    df["Amount"].values.reshape(-1, 1)
)

df["scaled_time"] = scaler.fit_transform(
    df["Time"].values.reshape(-1, 1)
)


# Remove original columns
df.drop(
    ["Time", "Amount"],
    axis=1,
    inplace=True
)

print("Scaling completed.")


# ============================================================
# 7. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop(
    "Class",
    axis=1
)

y = df["Class"]

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 9. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        class_weight="balanced",
        random_state=SEED
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1
    )
}


# ============================================================
# 10. TRAIN AND EVALUATE MODELS
# ============================================================

model_results = {}

plt.figure(figsize=(8, 6))

for name, model in models.items():

    print("\n")
    print("=" * 60)
    print("MODEL:", name)
    print("=" * 60)

    # Train
    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    # Predictions
    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Legitimate",
                "Fraud"
            ]
        )
    )


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(cm)


    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        f"\nROC-AUC Score: {auc:.4f}"
    )

    model_results[name] = auc


    # --------------------------------------------------------
    # ROC Curve
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )


# ============================================================
# 11. ROC CURVE
# ============================================================

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curves for Fraud Detection Models"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "model_evaluation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print("\nSaved: model_evaluation.png")


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)

for name, score in model_results.items():

    print(
        f"{name}: ROC-AUC = {score:.4f}"
    )


# ============================================================
# 13. BEST MODEL
# ============================================================

best_model = max(
    model_results,
    key=model_results.get
)

best_score = model_results[best_model]

print("\nBest Model:")
print(best_model)

print(
    f"Best ROC-AUC Score: {best_score:.4f}"
)


# ============================================================
# 14. COMPLETION MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated files:")
print("1. eda_overview.png")
print("2. model_evaluation.png")
