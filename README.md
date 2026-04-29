# Flood Probability Regression Project

This repository contains an applied machine learning project that predicts flood probability using environmental, infrastructure, and geographic features. The work demonstrates data exploration, preprocessing, model training, hyperparameter tuning, and feature selection for a regression problem.

## Project Overview

- **Goal:** Predict `FloodProbability` from features such as `MonsoonIntensity`, `TopographyDrainage`, `RiverManagement`, `Urbanization`, and related risk factors.
- **Dataset:** `flood.csv` contains 21 predictive columns and the target variable `FloodProbability`.
- **Approach:** Explore summary statistics, identify outliers, analyze correlations, scale data, and compare regression models.

## What’s Included

- `regression.ipynb` — a Jupyter notebook with the full workflow:
  - dataset loading and inspection
  - descriptive statistics and summary analysis
  - outlier visualization via boxplots
  - correlation matrix analysis
  - train/test split and scaling with `MinMaxScaler` and `RobustScaler`
  - model training for Linear Regression, Random Forest Regression, and MLP Regression
  - MLP hyperparameter tuning with `RandomizedSearchCV`
  - feature selection experiments on top-correlated predictors
- `summary.html` — an HTML summary output for dataset statistics.
- `flood.csv` — the raw dataset used for modeling.

## Key Findings

- The dataset columns show very similar summary statistics and correlation patterns across predictors.
- Both scaled and unscaled models achieved strong performance, with the `MLPRegressor` tuning experiment selected for better generalization.
- Feature selection experiments using top-correlated features were evaluated to understand model behavior on smaller predictor sets.

## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Reproduce

1. Open `regression.ipynb` in Jupyter Notebook or JupyterLab.
2. Run each cell sequentially to explore the data, train models, and evaluate results.
3. Review the comparison between scaling methods, model performance, and feature selection.

## Why This Project

This project is a practical example of regression modeling on a real-world-style dataset with many similar features. It highlights the full machine learning pipeline from data inspection through model tuning and offers a clean narrative
