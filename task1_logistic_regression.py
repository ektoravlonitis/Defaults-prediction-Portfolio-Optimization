# -*- coding: utf-8 -*-
"""
Created on Sat Mar 2 20:17:43 2024

@author: he_98
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Loading and preparing data
file_path = 'bankPortfolios.csv'

column_names = [
    "Loans for construction and land development", "Loans secured by farmland",
    "Loans secured by 1-4 family residential properties", "Loans secured by multi-family (> 5) residential properties",
    "Loans secured by non-farm non-residential properties", "Agricultural loans",
    "Commercial and industrial loans", "Loans to individuals",
    "All other loans (excluding consumer loans)", "Obligations (other than securities and leases) of states and political subdivision in the U.S.",
    "Held-to-maturity securities", "Available-for-sale securities, total",
    "Premises and fixed assets including capitalized lease", "Cash", "Banks debt", "Default"
]

# Functions for undersampling, PCA and evaluation metrics
def perform_undersampling(df):
    majority_class = df[df['Default'] == 0]
    minority_class = df[df['Default'] == 1]
    majority_class_undersampled = majority_class.sample(n=len(minority_class), random_state=42)
    return pd.concat([minority_class, majority_class_undersampled]).sample(frac=1, random_state=42)

def apply_pca(pca, X_train, X_test):
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    return X_train_pca, X_test_pca

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Metrics with {model_name}: Accuracy = {accuracy}, Precision = {precision}, Recall = {recall}, F1 Score = {f1}, ROC AUC = {roc_auc}")
    print(f"Confusion Matrix ({model_name}):")
    print(conf_matrix)


df = pd.read_csv(file_path, header=None, names=column_names)
balanced_df = perform_undersampling(df)

# Plotting the distribution of 'Default' and 'Non-Default' samples before undersampling
plt.figure(figsize=(6, 6))
sns.countplot(data=df, x='Default', palette={0: 'blue', 1: 'orange'})
plt.title('Distribution of Default and Non-Default Samples (Before Undersampling)')
plt.xlabel('Default')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Non-Default', 'Default'])
plt.show()

# Standardizing features
scaler = StandardScaler()
X = balanced_df.drop(['Default', 'Loans secured by 1-4 family residential properties', 'Banks debt'], axis=1)
y = balanced_df['Default']
X_standardized = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3, random_state=42)

# 1. Logistic Regression with Ridge Regularization
logreg_ridge = LogisticRegression(max_iter=1000)
logreg_ridge.fit(X_train, y_train)
evaluate_model(logreg_ridge, X_test, y_test, "Ridge")

# 2. Logistic Regression with PCA
pca = PCA(n_components=5)
X_train_pca, X_test_pca = apply_pca(pca, X_train, X_test)
logreg_pca = LogisticRegression(max_iter=1000)
logreg_pca.fit(X_train_pca, y_train)
evaluate_model(logreg_pca, X_test_pca, y_test, "PCA")

# 3. Logistic Regression without Regularization
logreg_none = LogisticRegression(penalty='none', max_iter=1000)
logreg_none.fit(X_train, y_train)
evaluate_model(logreg_none, X_test, y_test, "No Regularization")

# 4. Logistic Regression with Lasso Regularization
logreg_lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
logreg_lasso.fit(X_train, y_train)
evaluate_model(logreg_lasso, X_test, y_test, "Lasso")
