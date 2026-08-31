# CodSoft ML Internship - Task 2
# Credit Card Fraud Detection
# Santanu Mondal

# Dataset used: Credit Card Fraud Detection dataset from Kaggle
# https://www.kaggle.com/mlg-ulb/creditcardfraud
# It has transactions from European cardholders, most columns (V1-V28) are
# already PCA transformed so we can't really tell what they mean, only
# Time, Amount and Class (0 = normal, 1 = fraud) are original.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# just fixing this so results dont change every time i re run it
SEED = 42

print("loading dataset...")
df = pd.read_csv("creditcard.csv")
print(df.shape)
print(df["Class"].value_counts())

# fraud cases are super rare compared to normal ones, less than 1%
# so this is going to be an imbalanced classification problem
fraud_percent = df["Class"].mean() * 100
print(f"fraud is only {fraud_percent:.3f}% of the data, pretty imbalanced")

# quick look at class balance and amount for both classes
plt.figure(figsize=(11, 4))

plt.subplot(1, 2, 1)
sns.countplot(x="Class", data=df)
plt.yscale("log")
plt.title("legit (0) vs fraud (1) count")

plt.subplot(1, 2, 2)
sns.boxplot(x="Class", y="Amount", data=df)
plt.ylim(0, 300)  # clipping outliers so the box plot is actually readable
plt.title("amount by class")

plt.tight_layout()
plt.savefig("eda_overview.png")
plt.close()
print("saved eda_overview.png")

# Time and Amount need scaling since they're on a totally different scale
# compared to the V1-V28 columns which are already scaled from PCA
scaler = StandardScaler()
df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
df["scaled_time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))
df = df.drop(["Time", "Amount"], axis=1)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"train size: {len(X_train)}, test size: {len(X_test)}")

# using class_weight=balanced instead of undersampling/oversampling
# because we lose too much data with undersampling and oversampling
# felt like overkill for this assignment
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, class_weight="balanced", random_state=SEED),
}

plt.figure(figsize=(7, 6))

for name, model in models.items():
    print(f"\ntraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(name)
    # accuracy alone doesnt mean much here since dataset is so imbalanced
    # (predicting everything as legit would still give like 99.8% accuracy lol)
    # so precision/recall/f1 for the fraud class matter way more
    print(classification_report(y_test, preds, target_names=["legit", "fraud"]))
    print("confusion matrix:")
    print(confusion_matrix(y_test, preds))

    auc = roc_auc_score(y_test, probs)
    print(f"roc auc score: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (auc={auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves for all models")
plt.legend()
plt.savefig("model_evaluation.png")
plt.close()
print("\nsaved model_evaluation.png")
print("done")
