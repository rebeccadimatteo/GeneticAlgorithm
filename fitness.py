# fitness.py
import pandas as pd
import numpy as np
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from data_preparation import prepare_data_model


def fairness_metrics(test_data, test_indices, protected_attribute, predictions, target_column):
    """
    Calculate fairness metrics such as disparity, statistical parity, and equal opportunity.

    Args:
        test_data (pd.DataFrame): The test dataset.
        test_indices (pd.Index): Indices of the test samples.
        protected_attribute (str): The protected attribute to consider for fairness.
        predictions (array): Predictions made by the model.
        target_column (str): The target column name.

    Returns:
        dict: A dictionary containing the fairness metrics.
    """
    predictions_series = pd.Series(predictions, index=test_indices)
    positive_indices = test_indices[test_data.loc[test_indices, target_column] == 1]
    predictions_aligned = predictions_series.loc[positive_indices]
    actual_positive = test_data.loc[positive_indices, protected_attribute].value_counts(normalize=True)
    predicted_positive = predictions_aligned.value_counts(normalize=True)
    disparity = np.abs(actual_positive - predicted_positive).sum()
    predicted_positive_rate = predictions_aligned.value_counts(normalize=True)
    statistical_parity = np.abs(predicted_positive_rate - actual_positive).sum()
    true_positives = test_data[(test_data[target_column] == 1) & (predictions_series == 1)][
        protected_attribute].value_counts(normalize=True)
    opportunity_difference = np.abs(true_positives - actual_positive).sum()

    return {
        'disparity': disparity,
        'statistical_parity': statistical_parity,
        'equal_opportunity': opportunity_difference
    }


def fitness(data, techniques, models, protected_attribute, target_column):
    """
    Calculate the fitness value of a model given specific data preprocessing techniques and models.

    Args:
        data (pd.DataFrame): The dataset to be used.
        techniques (list): List of preprocessing techniques to apply.
        models (list): List of models to train.
        protected_attribute (str): The protected attribute for fairness evaluation.
        target_column (str): The target column name.

    Returns:
        float: The fitness value.
        dict: A dictionary containing various performance and fairness metrics.
    """
    data = prepare_data_model(data, target_column)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = {
        'logistic_regression': LogisticRegression(max_iter=1000, solver='liblinear'),
        'random_forest': RandomForestClassifier(),
        'svm': SVC(),
        'knn': KNeighborsClassifier(),
        'gradient_boosting': GradientBoostingClassifier()
    }

    # Apply each preprocessing technique
    for technique in techniques:
        if technique == 'onehot_standard':
            pass  # Apply onehot encoding and standardization (already done)
        elif technique == 'stratified_sampling':
            pass  # Apply stratified sampling
        elif technique == 'oversampling':
            pass  # Apply oversampling
        elif technique == 'undersampling':
            pass  # Apply undersampling
        elif technique == 'clustering':
            pass  # Apply clustering
        elif technique == 'ipw':
            pass  # Apply inverse propensity weighting
        elif technique == 'matching':
            pass  # Apply matching
        elif technique == 'min_max_scaling':
            pass  # Apply min-max scaling

    predictions = np.zeros(len(X_test))
    for model in models:
        classifier = classifiers[model]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        predictions += y_pred

    y_pred_final = (predictions / len(models)) > 0.5

    accuracy = accuracy_score(y_test, y_pred_final)
    precision = precision_score(y_test, y_pred_final, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred_final, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred_final, average='weighted')

    fairness = fairness_metrics(data, X_test.index, protected_attribute, y_pred_final, target_column)
    performance_score = (accuracy + precision + recall + f1) / 4
    fairness_score = sum(fairness.values())

    fitness_value = (1 - performance_score) + fairness_score

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        **fairness
    }

    return fitness_value, metrics


def fitness_model_optimization(model, techniques, X, y, protected_attribute, X_df, output_dir):
    """
    Optimize the fitness of a given model using various optimization techniques.

    Args:
        model: The model to be optimized.
        techniques (list): List of optimization techniques to apply.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        protected_attribute (str): The protected attribute for fairness evaluation.
        X_df (pd.DataFrame): The original data frame with all features.
        output_dir (str): Directory to save the optimized model.

    Returns:
        The best optimized model, its fitness value, and associated metrics.
    """
    from model_optimization import hyperparameter_tuning, outcomes_transformation, outcomes_optimization
    import joblib

    best_fitness = float('inf')
    best_model = None

    for technique in techniques:
        if technique == 'hyperparameter_tuning':
            current_model = hyperparameter_tuning(model, X, y, output_dir)
        elif technique == 'outcomes_transformation':
            current_model = outcomes_transformation(model, X, y, protected_attribute, output_dir, X_df)
        elif technique == 'outcomes_optimization':
            current_model = outcomes_optimization(model, X, y, protected_attribute, output_dir, X_df)
        else:
            continue

        y_pred = current_model.predict(X) if not isinstance(current_model,
                                                            ThresholdOptimizer) else current_model.predict(X,
                                                                                                           sensitive_features=
                                                                                                           X_df[
                                                                                                               protected_attribute])

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted')

        fairness = fairness_metrics(X_df, X.index, protected_attribute, y_pred, y.name)
        performance_score = (accuracy + precision + recall + f1) / 4
        fairness_score = sum(fairness.values())

        fitness_value = (1 - performance_score) + fairness_score

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            **fairness
        }

        print(f"Technique: {technique}")
        print(f"Performance metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        print(f"Fairness metrics: {fairness}")
        print(f"Fitness value: {fitness_value}")
        print(f"Individual: {technique}\n")

        if fitness_value < best_fitness:
            best_fitness = fitness_value
            best_model = current_model

    return best_model, best_fitness, metrics
