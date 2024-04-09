# -*- coding: utf-8 -*-
"""
Created on Sat Mar 2 20:17:43 2024

@author: he_98
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_curve, auc

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

# Function to create model for KerasClassifier
def create_model(layers=1, neurons=10, activation='relu'):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation=activation))
    for i in range(layers-1):  # Adding extra layers if layers > 1
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))  # Output layer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Creating the KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=0)

# Defining the grid search parameters for Grid Search
param_grid = {
    'layers': [1, 2, 3],
    'neurons': [5, 10, 15],
    'activation': ['relu', 'tanh'],
    'batch_size': [10, 20, 40],
    'epochs': [50, 100]
}

# Creating GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# Summarizing results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# After finding the best parameters, evaluating them on the test set
best_model = grid_result.best_estimator_

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Converting predictions from probabilities to binary
y_pred_bin = (y_pred > 0.5).astype(int)

# Calculating metrics
test_accuracy = accuracy_score(y_test, y_pred_bin)
test_precision = precision_score(y_test, y_pred_bin)
test_recall = recall_score(y_test, y_pred_bin)
test_f1 = f1_score(y_test, y_pred_bin)
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

print("Test set metrics:")
print(f"Accuracy: {test_accuracy}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1 Score: {test_f1}")
print(f"ROC AUC: {test_roc_auc}")

# Calculating ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plotting ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
