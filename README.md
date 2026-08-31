# Credit Card Fraud Detection - CodSoft ML Internship Task 2

This is my submission for Task 2 of the CodSoft Machine Learning virtual internship.

## What this does

Builds a model that can tell if a credit card transaction is fraud or not, based on transaction data. The tricky part with this kind of problem is that fraud cases are super rare (way less than 1% of all transactions), so a model that just predicts "not fraud" every single time would already get like 99.8% accuracy while being completely useless. Because of that I focused more on precision/recall/ROC-AUC instead of just accuracy.

## Dataset

Used the Credit Card Fraud Detection dataset from Kaggle:
https://www.kaggle.com/mlg-ulb/creditcardfraud

It's transactions made by European cardholders over 2 days in Sept 2013. Most of the features (V1 to V28) went through PCA already for privacy reasons so we don't actually know what they represent, only `Time`, `Amount` and `Class` are the original columns.

Didn't upload the dataset itself here since it's around 150mb, you'll need to download it from the Kaggle link if you want to run this.

## What I did

- Looked at the class imbalance first (barely any fraud cases compared to legit ones)
- Scaled Time and Amount since V1-V28 were already scaled from the PCA
- Tried three different models: Logistic Regression, Decision Tree, and Random Forest
- Used `class_weight="balanced"` on all of them so the models don't just ignore the minority (fraud) class
- Compared them using precision, recall, f1-score and ROC-AUC instead of plain accuracy

## Files

- `fraud_detection.py` - main code, run this after you have creditcard.csv
- `generate_sample_data.py` - makes fake data so you can test the code runs without needing to download the real dataset first
- `requirements.txt` - libraries needed

## How to run

```
pip install -r requirements.txt
python fraud_detection.py
```

(Make sure creditcard.csv is in the same folder, or run generate_sample_data.py first if you just want to test it)

## Results

Random Forest came out ahead in terms of ROC-AUC in my testing. Full output/plots get printed when you run the script (eda_overview.png and model_evaluation.png).

---
Santanu Mondal
B.Tech CSE (AI & ML), Adamas University

Done as part of the CodSoft Machine Learning internship, Sept 2026.
