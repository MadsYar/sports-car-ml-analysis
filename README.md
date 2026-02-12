# Sports Car Machine Learning Analysis

A machine learning project for analyzing sports car data, including classification and regression tasks using various ML algorithms and techniques.

## Author

Course Project - Introduction to Machine Learning and Data Mining (Technical University of Denmark)

## Project Overview

This project applies multiple machine learning techniques to a sports car dataset to predict:
- **Classification**: Country of origin of sports cars
- **Regression**: Price of sports cars based on performance characteristics

The project implements and compares various ML algorithms including logistic regression, K-nearest neighbors (KNN), neural networks and regularized linear regression with proper cross-validation and statistical analysis.

## Dataset

**Dataset**: `carz.csv`
- **Size**: 130+ luxury sports car records
- **Features**:
  - Car Make (e.g., Porsche, Ferrari, BMW)
  - Country (e.g., Germany, Italy, USA, Japan, UK)
  - Year (2015-2022)
  - Horsepower
  - Torque (lb-ft)
  - 0-60 MPH Time (seconds)
  - Price (USD)

## Project Structure

### Core Modules

- **`Extract.py`**: Data loading and preprocessing
  - Loads CSV data using Pandas
  - Encodes categorical variables (Car Make, Country) as numeric labels
  - Prepares feature matrix (X) and target variables (y)

- **`Classification.py`**: Exploratory classification visualization
  - Visualizes sports cars in 2D feature space
  - Colored by country of origin
  - Helps understand data distribution and separability

- **`Class-2.py`**: Multinomial logistic regression classifier
  - Implements regularized logistic regression
  - 10-fold outer cross-validation with 5-fold inner validation
  - Lambda regularization parameter tuning
  - Computes training/test error and R² metrics

- **`KNNBUTBETTER.py`**: K-nearest neighbors classifier
  - Grid search over k values (1-50)
  - Nested cross-validation for hyperparameter tuning
  - Standardization of features
  - Generates predictions for statistical comparison

- **`Baseline.py`**: Baseline classification model
  - Most frequent class classifier per fold
  - Establishes baseline performance metric
  - Useful for benchmarking other models

- **`PCA.py`**: Principal Component Analysis
  - Computes PCA using SVD decomposition
  - Visualizes variance explained by principal components
  - Shows data projection onto principal component space
  - Displays component loadings

- **`Regression_part_A.py`**: Linear regression for price prediction
  - Regularized Linear Regression (RLR) and standard linear regression
  - Predicts car price from Horsepower, Torque, and 0-60 MPH Time
  - 10-fold cross-validation
  - Compares with baseline (mean predictor)
  - MSE and R² performance metrics

- **`Regression_part_B.py`**: Advanced regression comparison
  - Compares Ridge regression vs. Artificial Neural Networks (ANN)
  - Nested 10-fold cross-validation
  - Implements PyTorch neural networks with configurable hidden units
  - Statistical hypothesis testing (paired t-tests, confidence intervals)
  - McNemar test for model comparison

- **`stats.py`**: Statistical model comparison
  - Imports predictions from all models
  - Performs McNemar's test for pairwise model comparisons
  - Compares:
    - KNN vs. Logistic Regression
    - KNN vs. Baseline
    - Logistic Regression vs. Baseline
  - Confidence intervals and p-values at 5% significance level

- **`toolbox_02450/`**: DTU Machine Learning Toolbox
  - Contains utility functions for ML operations
  - Regularized linear regression validation
  - Neural network training and visualization tools

## Notes

- Path references in some files may need adjustment based on local system
- The DTU 02450 Toolbox functions are used for regularized linear regression validation
- Some scripts use hardcoded paths that should be updated to match your environment
- Cross-validation uses `shuffle=False` in some CV instances for reproducibility
