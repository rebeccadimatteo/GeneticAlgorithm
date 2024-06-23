# genetic_algorithm.py
import numpy as np
from fitness import fitness, fitness_model_optimization
from data_preparation import prepare_data_for_model_optimization
import os
import joblib


def genetic_algorithm(operation, dataset, protected_attribute, target_column, model_info=None, generations=10,
                      population_size=10):
    """
    Run a genetic algorithm to optimize either a dataset or a model.

    Args:
        operation (str): 'dataset' or 'model', indicating what to optimize.
        dataset (pd.DataFrame): The dataset to optimize.
        protected_attribute (str): The name of the protected attribute column.
        target_column (str): The name of the target column.
        model_info (tuple): Additional model info if optimizing a model. Contains model, output_dir, X, y, X_df.
        generations (int): Number of generations to run the algorithm.
        population_size (int): Size of the population in each generation.

    Returns:
        tuple: Best solution found by the genetic algorithm.
    """

    # Define potential techniques and models for optimization
    techniques = [
        'onehot_standard', 'stratified_sampling', 'oversampling', 'undersampling',
        'clustering', 'ipw', 'matching', 'min_max_scaling'
    ]
    models = [
        'logistic_regression', 'random_forest', 'svm', 'knn', 'gradient_boosting'
    ]

    model_optimization_techniques = [
        'hyperparameter_tuning', 'outcomes_transformation', 'outcomes_optimization'
    ]

    # Initialize population based on the operation type
    if operation == 'dataset':
        # Generate random pairs of techniques and models for the initial population
        population = [(np.random.choice(techniques), np.random.choice(models)) for _ in range(population_size)]
    else:
        # Generate random model optimization techniques for the initial population
        population = [np.random.choice(model_optimization_techniques) for _ in range(population_size)]

    best_solution = None

    # Iterate through each generation
    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations} start")

        if operation == 'dataset':
            # Evaluate fitness of each (technique, model) pair in the population
            fitness_scores = [
                (technique, model, fitness(dataset.copy(), technique, model, protected_attribute, target_column))
                for (technique, model) in population
            ]
        else:
            model, output_dir, X, y, X_df = model_info

            # Ensure the output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Evaluate fitness of each technique in the population
            fitness_scores = [
                (technique,
                 fitness_model_optimization(model, [technique], X, y, protected_attribute, output_dir, X_df)[1])
                for technique in population
            ]

        # Sort the population by fitness scores
        sorted_population = sorted(fitness_scores, key=lambda x: x[2] if operation == 'dataset' else x[1])
        # Select the top half of the population for breeding
        best_individuals = sorted_population[:population_size // 2]

        if operation == 'dataset':
            best_techniques = [ind[0] for ind in best_individuals]
            best_models = [ind[1] for ind in best_individuals]

            next_generation = []
            # Create the next generation by combining techniques and models from the best individuals
            while len(next_generation) < population_size:
                parent_techniques = np.random.choice(best_techniques, 2, replace=False)
                parent_models = np.random.choice(best_models, 2, replace=False)
                child_technique = np.random.choice(parent_techniques)
                child_model = np.random.choice(parent_models)

                # Apply mutation with a small probability
                if np.random.rand() < 0.1:
                    child_technique = np.random.choice(techniques)
                if np.random.rand() < 0.1:
                    child_model = np.random.choice(models)

                next_generation.append((child_technique, child_model))
            population = next_generation
        else:
            best_techniques = [ind[0] for ind in best_individuals]

            next_generation = []
            # Create the next generation by combining techniques from the best individuals
            while len(next_generation) < population_size:
                parent_techniques = np.random.choice(best_techniques, 2, replace=False)
                child_technique = np.random.choice(parent_techniques)

                # Apply mutation with a small probability
                if np.random.rand() < 0.1:
                    child_technique = np.random.choice(model_optimization_techniques)

                next_generation.append(child_technique)
            population = next_generation

        print(f"End of generation {generation + 1}")

    # Select the best solution based on the lowest fitness score
    best_solution = min(fitness_scores, key=lambda x: x[2] if operation == 'dataset' else x[1])
    if operation == 'dataset':
        print(f"Best solution: Technique={best_solution[0]}, Model={best_solution[1]}, Fitness={best_solution[2]}")
        return best_solution
    else:
        print(f"Best solution: Technique={best_solution[0]}, Fitness={best_solution[1]}")
        return best_solution
