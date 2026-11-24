# Homework 3 — Income Prediction (Logistic Regression & Random Forest)

Course **DS07 Supervised Learning: Classification**. The goal is to predict whether a person's annual income is **above 50K USD**, using the UCI Adult census dataset (`adult.csv`).

## Files

| File | Description |
|---|---|
| `Homework3_Income_Prediction.ipynb` | Main notebook. It contains every step with explanations, figures, comments and all outputs already run. |
| `adult.csv` | Dataset: 48,842 records, 15 columns. |

## How to run (Google Colab)

1. Open `Homework3_Income_Prediction.ipynb` in Google Colab.
2. Click **Runtime → Run all**.
3. When the upload cell runs, click **Choose Files** and select `adult.csv` from your computer.

The whole notebook takes about **5–8 minutes** on Colab, most of it in the Random Forest grid search. It was tested with Python 3.13 and works on scikit-learn versions before and after 1.8. Outside Colab, it reads `adult.csv` from the working directory.

## Steps (as required by the assignment)

1. **Exploratory data analysis:** structure, missing values (`?`), duplicates, class balance, numerical and categorical features against income, and correlations.
2. **Target encoding:** `<=50K` → 0, `>50K` → 1.
3. **Train/test split:** 80% / 20%, stratified, `random_state=42`.
4. **Logistic Regression with L1 and L2 regularization:** `C` is tuned with 5-fold stratified CV inside a leak-free preprocessing pipeline (log transform, scaling, one-hot encoding).
5. **Random Forest:** a default model, then a grid search over `max_depth`, `min_samples_leaf` and `max_features`.
6. **Test-set evaluation:** classification reports, confusion matrices, ROC and PR curves, decision-threshold optimisation (chosen on training CV only), permutation importance and a final comparison.

## Main results (test set, 9,758 people)

| Model | Accuracy | F1 (>50K) | ROC-AUC |
|---|---|---|---|
| Logistic Regression (L1) | 0.842 | 0.638 | 0.896 |
| Logistic Regression (L2) | 0.843 | 0.640 | 0.896 |
| Random Forest (tuned) | **0.864** | 0.681 | **0.918** |
| Random Forest (tuned) + optimised threshold (0.39) | 0.858 | **0.711** | **0.918** |

For reference, a model that always predicts `<=50K` gets an accuracy of 0.761.
