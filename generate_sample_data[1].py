# quick script to make some fake transaction data so I could test my
# code before the real Kaggle dataset finished downloading
# (real dataset is like 150mb so it takes a bit)
# NOT the actual data, just for testing the pipeline runs without errors

import numpy as np
import pandas as pd

np.random.seed(42)

n = 5000
fraud_ratio = 0.017
n_fraud = int(n * fraud_ratio)
n_legit = n - n_fraud

data = {"Time": np.random.uniform(0, 172800, n)}

for i in range(1, 29):
    legit_col = np.random.normal(0, 1, n_legit)
    fraud_col = np.random.normal(0.6, 1.3, n_fraud)
    data[f"V{i}"] = np.concatenate([legit_col, fraud_col])

data["Amount"] = np.concatenate([
    np.random.exponential(60, n_legit),
    np.random.exponential(120, n_fraud),
])
data["Class"] = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("creditcard.csv", index=False)

print(f"made a fake dataset with {len(df)} rows, {int(df['Class'].sum())} fraud")
print("remember this is fake data just to test the code works")
