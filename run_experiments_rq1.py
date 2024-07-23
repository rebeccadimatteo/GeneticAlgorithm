import pandas as pd
import os
import time
from genetic_algorithm import genetic_algorithm
from data_preparation import sample_dataset, prepare_data_dataset, prepare_data_for_model_optimization

# Funzione per salvare i risultati in un file Excel
def save_results_to_excel(results, file_path):
    results_df = pd.DataFrame(results)
    results_df.to_excel(file_path, index=False)
    print(f"Results saved to {file_path}")

# Funzione per eseguire l'algoritmo genetico con parametri specificati
def run_genetic_algorithm(operation, dataset, protected_attribute, target_column, additional_info, generations, population_size):
    start_time = time.time()

    best_solution = genetic_algorithm(operation, dataset, protected_attribute, target_column, additional_info,
                                      generations=generations, population_size=population_size)

    end_time = time.time()
    execution_time = end_time - start_time

    if operation == 'dataset':
        techniques, models, fitness_value, metrics = best_solution
        energy_consumption = metrics.get('energy', 0)  # Assumendo che ci sia una metrica di consumo energetico
        accuracy = metrics.get('accuracy', 0)
        fairness = metrics.get('fairness', 0)
    else:
        techniques, fitness_value, metrics = best_solution
        energy_consumption = metrics.get('energy', 0)  # Assumendo che ci sia una metrica di consumo energetico
        accuracy = metrics.get('accuracy', 0)
        fairness = metrics.get('fairness', 0)

    return execution_time, energy_consumption, accuracy, fairness

# Funzione per eseguire gli esperimenti
def execute_genetic_algorithm_experiments(operation, dataset, protected_attribute, target_column, additional_info, sample_fraction, file_name):
    results = []
    combinations = [
        (10, 5, 0.01),
        (20, 10, 0.01),
        (30, 15, 0.02),
        (40, 20, 0.02),
        (50, 25, 0.03),
        (50, 50, 0.03),   # Nuova combinazione
        (100, 50, 0.03),  # Nuova combinazione
        (100, 100, 0.03)  # Nuova combinazione
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

# Imposta i parametri direttamente nel codice
operation = 'dataset'  # oppure 'model'
dataset_path = 'Dataset/dataset1.csv'
protected_attribute = 'Sex_Code_Text'
target_column = 'DecileScore'
output_dir = 'results'
sample_fraction = 0.1  # ad esempio

# Crea la cartella dei risultati se non esiste
os.makedirs(output_dir, exist_ok=True)

# Carica il dataset
dataset = pd.read_csv(dataset_path)

# Esempio per ottimizzazione del dataset
dataset_sample = sample_dataset(dataset, fraction=sample_fraction)
dataset_sample = prepare_data_dataset(dataset_sample, target_column)

# Esegui gli esperimenti
results = execute_genetic_algorithm_experiments(operation, dataset_sample, protected_attribute, target_column, output_dir, sample_fraction, 'experiment_results')

# Salva i risultati nella cartella 'results'
results_file_path = os.path.join(output_dir, 'experiments_rq1_results.xlsx')
save_results_to_excel(results, results_file_path)
print(f"Results file saved at: {results_file_path}")
                                                                                                                  