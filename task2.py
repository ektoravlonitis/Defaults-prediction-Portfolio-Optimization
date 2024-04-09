# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 14:29:26 2024

@author: he_98
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np

# Loading the dataset
df = pd.read_csv('48_Industry_Portfolios_daily.csv', index_col=0)
df.index.name = 'Date'

matching_indices = [i for i, idx in enumerate(df.index.astype(str)) if 'Average Equal Weighted Returns -- Daily' in idx]

title_row_index = matching_indices[0]
print(f"Title row found at index: {title_row_index}")
    
# Splitting the DataFrame based on this index
df_first = df.iloc[:title_row_index, :]
df_second = df.iloc[title_row_index + 2:, :]

df_first.index = pd.to_datetime(df_first.index, format='%Y%m%d')
df_second.index = pd.to_datetime(df_second.index, format='%Y%m%d')

# Choosing the first dataset

print(df_first.dtypes)
# Converting columns to numeric
df_first = df_first.apply(pd.to_numeric, errors='coerce')

print(df_first.isnull().sum())
# Filling NaN values with forward fill method
df_first.fillna(method='ffill', inplace=True)

# Next, the covariance matrix is calculated
covariance_matrix = df_first.cov()

# The portfolio variance equation:
def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

num_assets = len(df_first.columns)
initial_guess = np.repeat(1/num_assets, num_assets)
constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

# Minimizing variance
result = minimize(portfolio_variance, initial_guess, args=(covariance_matrix,), method='SLSQP', constraints=constraints, bounds=[(-1, 1)]*num_assets)
optimal_weights = result.x

# Different time windows
time_windows = [5, 10, 20]

optimal_weights_time = {}

for window in time_windows:
    # Selecting subset of df_first based on the time window
    end_date = df_first.index.max()
    start_date = end_date - pd.DateOffset(years=window)
    df_subset = df_first.loc[start_date:end_date]
    
    # Calculating covariance matrix for the subset
    covariance_matrix_subset = df_subset.cov()
    
    # Optimization for the subset
    result_subset = minimize(portfolio_variance, initial_guess, args=(covariance_matrix_subset,), method='SLSQP', constraints=constraints, bounds=[(-1, 1)]*num_assets)
    
    # Storing optimal weights for the time window
    optimal_weights_time[window] = result_subset.x

for window, weights in optimal_weights_time.items():
    print(f"Optimal weights for {window}-year window: {weights}")
    subset_covariance = df_first.loc[end_date - pd.DateOffset(years=window):end_date].cov()
    variance = portfolio_variance(weights, subset_covariance)
    print(f"Total portfolio variance for {window}-year window: {variance}")

# REGULARIZATION part of the analysis

# Hyperparameter Tuning

split_index = int(len(df_first) * 0.8)

# Splitting the data
df_train = df_first.iloc[:split_index]
df_validation = df_first.iloc[split_index:]

# Computing the covariance matrix for the training data
cov_matrix_train = df_train.cov()

def evaluate_portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

# Storing results for analysis
results = {}
validation_variances = []

for lambda_reg in np.logspace(-4, 1, 6):
    # Redefining the regularized variance function with current lambda_reg
    def portfolio_variance_reg(weights, cov_matrix):
        return weights.T @ cov_matrix @ weights + lambda_reg * np.sum(np.square(weights))
    
    # Minimizing regularized variance for the training set
    result = minimize(portfolio_variance_reg, initial_guess, args=(cov_matrix_train,), method='SLSQP', constraints=constraints, bounds=[(-1, 1)]*num_assets)
    
    # Storing the optimal weights
    results[lambda_reg] = result.x
    
    # Evaluating the performance on the validation set
    validation_variance = evaluate_portfolio_variance(result.x, df_validation.cov())
    validation_variances.append(validation_variance)

# Selecting the lambda_reg with the lowest variance on the validation set
best_lambda = list(np.logspace(-4, 1, 6))[validation_variances.index(min(validation_variances))]
best_weights = results[best_lambda]

print("best lambda: ", best_lambda)
print("best weights: ", best_weights)


lambda_reg = best_lambda  # Regularization parameter

# The portfolio variance equation with regularization:
def portfolio_variance_reg(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights + lambda_reg * np.sum(weights**2)

# Minimizing regularized variance
result_reg = minimize(portfolio_variance_reg, initial_guess, args=(covariance_matrix,), method='SLSQP', constraints=constraints, bounds=[(-1, 1)]*num_assets)
optimal_weights_reg = result_reg.x

# Ban on short selling

# Minimizing variance with short selling
result = minimize(portfolio_variance, initial_guess, args=(covariance_matrix,), method='SLSQP', constraints=constraints, bounds=[(-1, 1)]*num_assets)
print("result: ", result)

# Minimizing variance with no short selling
result_no_short = minimize(portfolio_variance, initial_guess, args=(covariance_matrix,), method='SLSQP', constraints=constraints, bounds=[(0, 1)]*num_assets)
optimal_weights_no_short = result_no_short.x

print("result_no_short: ", result_no_short)
print("optimal_weights_no_short: ", optimal_weights_no_short)

# Sampling optimal weights for visualization
optimal_weights_samples = {
    "Original": optimal_weights,
    "Regularized": optimal_weights_reg,
    "No Short Selling": optimal_weights_no_short
}

# Generating bar charts for each optimization strategy separately
asset_labels = df_first.columns.tolist()

# Original Optimization Strategy
plt.figure(figsize=(10, 6))
plt.bar(asset_labels, optimal_weights_samples["Original"], 
        color=['red' if w < 0 else 'green' for w in optimal_weights_samples["Original"]])
plt.title('Original Optimization Strategy')
plt.ylabel('Weight')
plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=90, fontsize=10) 
plt.show()

# Regularized Optimization Strategy
plt.figure(figsize=(10, 6))
plt.bar(asset_labels, optimal_weights_samples["Regularized"], 
        color=['red' if w < 0 else 'green' for w in optimal_weights_samples["Regularized"]])
plt.title('Regularized Optimization Strategy')
plt.ylabel('Weight')
plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=90, fontsize=10)
plt.show()

# No Short Selling Optimization Strategy
plt.figure(figsize=(10, 6))
plt.bar(asset_labels, optimal_weights_samples["No Short Selling"], 
        color=['red' if w < 0 else 'green' for w in optimal_weights_samples["No Short Selling"]])
plt.title('No Short Selling Optimization Strategy')
plt.ylabel('Weight')
plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=90, fontsize=10)
plt.show()

