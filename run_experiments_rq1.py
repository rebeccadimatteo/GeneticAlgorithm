import pandas as pd
import os
import time
from genetic_algorithm import genetic_algorithm
from data_preparation import sample_dataset, prepare_data_dataset, prepare_data_for_model_optimization

# Function to save results to an Excel file
def save_results_to_excel(results, file_path):
    results_df = pd.DataFrame(results)
    results_df.to_excel(file_path, index=False)
    print(f"Results saved to {file_path}")

# Function to run the genetic algorithm with specified parameters
def run_genetic_algorithm(operation, dataset, protected_attribute, target_column, additional_info, generations, population_size):
    start_time = time.time()

    best_solution = genetic_algorithm(operation, dataset, protected_attribute, target_column, additional_info,
                                      generations=generations, population_size=population_size)

    end_time = time.time()
    execution_time = end_time - start_time

    if operation == 'dataset':
        techniques, models, fitness_value, metrics = best_solution
        energy_consumption = metrics.get('energy', 0)  # Assuming there is an energy consumption metric
        accuracy = metrics.get('accuracy', 0)
        fairness = metrics.get('fairness', 0)
    else:
        techniques, fitness_value, metrics = best_solution
        energy_consumption = metrics.get('energy', 0)  # Assuming there is an energy consumption metric
        accuracy = metrics.get('accuracy', 0)
        fairness = metrics.get('fairness', 0)

    return execution_time, energy_consumption, accuracy, fairness

# Function to run the experiments
def execute_genetic_algorithm_experiments(operation, dataset, protected_attribute, target_column, additional_info, sample_fraction, file_name):
    results = []
    combinations = [
        (10, 5, 0.01),
        (20, 10, 0.01),
        (30, 15, 0.02),
        (40, 20, 0.02),
        (50, 25, 0.03),
        (50, 50, 0.03),   # New combination
        (100, 50, 0.03),  # New combination
        (100, 100, 0.03)  # New combination
    ]

    for population_size, generation, mutation_rate in combinations:
        execution_time, energy_consumption, accuracy, fairness = run_genetic_algorithm(
            operation, dataset, protected_attribute, target_column, additional_info, generation, population_size)

        results.append({
            "Population Size": population_size,
            "Generations": generation,
            "Mutation Rate": mutation_rate,
            "Execution Time (s)": execution_time,
            "Energy Consumption (kWh)": energy_consumption,
            "Accuracy": accuracy,
            "Fairness": fairness
        })

    return results

# Set the parameters directly in the code
operation = 'dataset'  # or 'model'
dataset_path = 'Dataset/dataset1.csv'
protected_attribute = 'Sex_Code_Text'
target_column = 'DecileScore'
output_dir = 'results'
sample_fraction = 0.1  # for example

# Create the results directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
dataset = pd.read_csv(dataset_path)

# Example for dataset optimization
dataset_sample = sample_dataset(dataset, fraction=sample_fraction)
dataset_sample = prepare_data_dataset(dataset_sample, target_column)

# Run the experiments
results = execute_genetic_algorithm_experiments(operation, dataset_sample, protected_attribute, target_column, output_dir, sample_fraction, 'experiment_results')

# Save the results in the 'results' directory
results_file_path = os.path.join(output_dir, 'experiments_rq1_results.xlsx')
save_results_to_excel(results, results_file_path)
print(f"Results file saved at: {results_file_path}")
