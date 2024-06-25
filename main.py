# main.py
import pandas as pd
from genetic_algorithm import genetic_algorithm
from data_preparation import sample_dataset, prepare_data_dataset, prepare_data_for_model_optimization
import joblib
import os

def get_user_input():
    """
    Get user input to determine if optimizing a dataset or a model.
    For datasets: specify file path, protected attribute, target variable, sample fraction, and output directory.
    For models: specify model path, dataset path, protected attribute, target variable, sample fraction, and output directory.

    Returns:
        tuple: Operation type, dataset, protected attribute, target column, additional info, sample fraction.
    """
    operation = input("Do you want to optimize a dataset or a model? Enter 'dataset' or 'model': ").strip().lower()

    if operation == 'dataset':
        # Get dataset path and load the dataset
        dataset_path = input("Enter the dataset path (e.g., 'Dataset/dataset1.csv'): ").strip()
        try:
            dataset = pd.read_csv(dataset_path)
            print("Loaded dataset with columns:", dataset.columns.tolist())
        except FileNotFoundError:
            print(f"The file {dataset_path} was not found. Please ensure the path is correct.")
            return None, None, None, None, None, None

        # Get the base name of the dataset file
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

        # Get protected attribute and target variable
        protected_attribute = input("Enter the name of the protected attribute (e.g., 'Sex_Code_Text'): ").strip()
        if protected_attribute not in dataset.columns:
            print(f"The protected attribute '{protected_attribute}' is not present in the dataset.")
            return None, None, None, None, None, None

        target_column = input("Enter the name of the target variable (e.g., 'DecileScore'): ").strip()
        if target_column not in dataset.columns:
            print(f"The target variable '{target_column}' is not present in the dataset.")
            return None, None, None, None, None, None

        # Get sample fraction and output directory for saving the optimized dataset
        sample_fraction = float(input("Enter the fraction of the dataset to use (e.g., 0.1 for 10%): ").strip())
        output_dir = input("Enter the output directory to save the optimized dataset (e.g., 'Output/'): ").strip()

        return 'dataset', dataset, protected_attribute, target_column, output_dir, sample_fraction, dataset_name

    elif operation == 'model':
        # Get model path and load the model
        model_path = input("Enter the trained model path (e.g., 'Models/trained_model.pkl'): ").strip()
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"The file {model_path} was not found. Please ensure the path is correct.")
            return None, None, None, None, None, None

        # Get the base name of the model file
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        # Get dataset path and load the dataset
        dataset_path = input("Enter the dataset path (e.g., 'Dataset/dataset1.csv'): ").strip()
        try:
            dataset = pd.read_csv(dataset_path)
        except FileNotFoundError:
            print(f"The file {dataset_path} was not found. Please ensure the path is correct.")
            return None, None, None, None, None, None

        print("The available columns in the dataset are:")
        print(dataset.columns.tolist())

        # Get protected attribute and target variable
        protected_attribute = input("Enter the name of the protected attribute (e.g., 'Sex_Code_Text'): ").strip()
        if protected_attribute not in dataset.columns:
            print(f"The protected attribute '{protected_attribute}' is not present in the dataset.")
            return None, None, None, None, None, None

        target_column = input("Enter the name of the target variable (e.g., 'DecileScore'): ").strip()
        if target_column not in dataset.columns:
            print(f"The target variable '{target_column}' is not present in the dataset.")
            return None, None, None, None, None, None

        # Get output directory for saving the optimized model
        output_dir = input("Enter the output directory to save the optimized model (e.g., 'Output/'): ").strip()

        sample_fraction = float(input("Enter the fraction of the dataset to use (e.g., 0.1 for 10%): ").strip())

        return 'model', dataset, protected_attribute, target_column, (model, output_dir), sample_fraction, model_name
    else:
        print("Invalid operation. Please enter 'dataset' or 'model'.")
        return None, None, None, None, None, None, None


# Get user input for dataset or model optimization
operation, dataset, protected_attribute, target_column, additional_info, sample_fraction, file_name = get_user_input()
if dataset is None:
    exit()

# Sample the dataset
dataset_sample = sample_dataset(dataset, fraction=sample_fraction)

# Prepare the data if optimizing the dataset
if operation == 'dataset':
    dataset_sample = prepare_data_dataset(dataset_sample, target_column)

# Prepare the data for model optimization if needed
if operation == 'model':
    X, y, dataset_sample = prepare_data_for_model_optimization(dataset_sample, target_column, protected_attribute)
    additional_info = (additional_info[0], additional_info[1], X, y, dataset_sample)

# Run the genetic algorithm for the selected operation
best_solution = genetic_algorithm(operation, dataset_sample, protected_attribute, target_column, additional_info,
                                  generations=2, population_size=5)

# Print and save the best solution found by the genetic algorithm
if operation == 'dataset':
    print(
        f"Best solution for dataset: Technique={best_solution[0]}, Model={best_solution[1]}, Fitness={best_solution[2]}")
    best_dataset_path = os.path.join(additional_info, f'best_optimized_dataset_{file_name}.csv')
    dataset_sample.to_csv(best_dataset_path, index=False)
    print(f"Optimized dataset saved at: {best_dataset_path}")
else:
    print(f"Best solution for model: Technique={best_solution[0]}, Fitness={best_solution[1]}")
    best_model_path = os.path.join(additional_info[1], f'best_model_{file_name}.pkl')
    joblib.dump(best_solution[0], best_model_path)
    print(f"Best model saved at: {best_model_path}")
