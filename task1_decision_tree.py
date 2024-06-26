# -*- coding: utf-8 -*-
"""
Created on Sat Mar 2 20:17:43 2024

@author: he_98
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree

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

# Standardizing features
scaler = StandardScaler()
X = balanced_df.drop(['Default', 'Loans secured by 1-4 family residential properties', 'Banks debt'], axis=1)
y = balanced_df['Default']
X_standardized = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3, random_state=42)

# Defining the parameter grid for the Grid Search
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['auto', 'sqrt', 'log2', None],
}

# Initializing the decision tree classifier
dt = DecisionTreeClassifier(random_state=42)

# Initializing the GridSearchCV object
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')

# Fitting it to the data
grid_search.fit(X_train, y_train)

# The best parameters and the corresponding score:
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# 1. Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
evaluate_model(dt_model, X_test, y_test, "Decision Tree")

# 2. Decision Tree with best parameters
best_dt = grid_search.best_estimator_
evaluate_model(best_dt, X_test, y_test, "Decision Tree with the best parameters")

# Plotting the tree.
plt.figure(figsize=(20,10))
plot_tree(best_dt, filled=True, feature_names=X.columns, class_names=["Non-Default", "Default"])
plt.title("Decision Tree with the Best Parameters")
plt.show()

# 3. Decision Tree with PCA
pca = PCA(n_components=5)
X_train_pca, X_test_pca = apply_pca(pca, X_train, X_test)
dt_pca = DecisionTreeClassifier(random_state=42)
dt_pca.fit(X_train_pca, y_train)
evaluate_model(dt_pca, X_test_pca, y_test, "Decision Tree with PCA")
