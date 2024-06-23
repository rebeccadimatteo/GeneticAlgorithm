# model_optimization.py
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
import joblib
import os
import pandas as pd

from data_preparation import preprocess_protected_attribute

def hyperparameter_tuning(model, X, y, output_dir):
    """
    Perform hyperparameter tuning using GridSearchCV to find the best model parameters.

    Args:
        model: The machine learning model to be optimized.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.

    Returns:
        The best model after hyperparameter tuning.
    """
    # Define a parameter grid for tuning hyperparameters
    param_grid = {'C': [0.01, 0.1, 1, 10]}
    # Use accuracy score as the evaluation metric
    scorer = make_scorer(accuracy_score)
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, scoring=scorer, cv=5)
    # Fit the grid search on the data
    grid_search.fit(X, y)
    # Retrieve the best model from grid search
    best_model = grid_search.best_estimator_
    return best_model  # Return the best model instead of saving it directly

def outcomes_transformation(model, X, y, protected_attribute, output_dir, X_df):
    """
    Apply outcomes transformation using ThresholdOptimizer to adjust model predictions
    to achieve demographic parity.

    Args:
        model: The machine learning model to be optimized.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        protected_attribute (str): The name of the protected attribute column.
        X_df (pd.DataFrame): DataFrame containing features and the protected attribute.

    Returns:
        The threshold optimizer model.
    """
    # Initialize ThresholdOptimizer for demographic parity constraint
    threshold_optimizer = ThresholdOptimizer(
        estimator=model,
        constraints="demographic_parity",
        prefit=False,
        predict_method="auto"
    )

    # Ensure the protected attribute exists in the DataFrame
    if protected_attribute not in X_df.columns:
        raise ValueError(f"Protected attribute '{protected_attribute}' not found in dataset columns.")

    # Preprocess the protected attribute to ensure it's suitable for optimization
    X_df[protected_attribute] = preprocess_protected_attribute(X_df, protected_attribute)

    # Fit the threshold optimizer on the data
    threshold_optimizer.fit(X, y, sensitive_features=X_df[protected_attribute])
    return threshold_optimizer  # Return the threshold optimizer instead of saving it directly

def outcomes_optimization(model, X, y, protected_attribute, output_dir,X_df):
    """
    Apply outcomes optimization using ThresholdOptimizer to adjust model predictions
    to achieve equalized odds.

    Args:
        model: The machine learning model to be optimized.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        protected_attribute (str): The name of the protected attribute column.
        X_df (pd.DataFrame): DataFrame containing features and the protected attribute.

    Returns:
        The threshold optimizer model.
    """
    # Initialize ThresholdOptimizer for equalized odds constraint
    threshold_optimizer = ThresholdOptimizer(
        estimator=model,
        constraints="equalized_odds",
        prefit=False,
        predict_method="auto"
    )

    # Ensure the protected attribute exists in the DataFrame
    if protected_attribute not in X_df.columns:
        raise ValueError(f"Protected attribute '{protected_attribute}' not found in dataset columns.")

    # Preprocess the protected attribute to ensure it's suitable for optimization
    X_df[protected_attribute] = preprocess_protected_attribute(X_df, protected_attribute)

    # Fit the threshold optimizer on the data
    threshold_optimizer.fit(X, y, sensitive_features=X_df[protected_attribute])
    return threshold_optimizer  # Return the threshold optimizer instead of saving it directly


