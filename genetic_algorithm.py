# genetic_algorithm.py
import numpy as np
from fitness import fitness, fitness_model_optimization
from data_preparation import prepare_data_for_model_optimization
import os
import joblib


def genetic_algorithm(operation, dataset, protected_attribute, target_column, model_info=None, generations=10,
                      population_size=10):
    """
    Implements a genetic algorithm for optimizing either dataset preprocessing or model hyperparameters.

    Args:
        operation (str): The type of operation to optimize ('dataset' or 'model').
        dataset (pd.DataFrame): The dataset to use.
        protected_attribute (str): The protected attribute for fairness evaluation.
        target_column (str): The target column name.
        model_info (tuple): Information about the model (for model optimization).
        generations (int): The number of generations to evolve.
        population_size (int): The size of the population.

    Returns:
        The best solution found by the genetic algorithm.
    """

    # List of techniques and models
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

    # Initialize population
    if operation == 'dataset':
        # Each individual is a combination of 8 preprocessing techniques and 3 models
        population = [
            (np.random.choice(techniques, size=8, replace=False), np.random.choice(models, size=3, replace=False)) for _
            in range(population_size)]
    else:
        # Each individual is a combination of 3 model optimization techniques
        population = [np.random.choice(model_optimization_techniques, size=3, replace=False) for _ in
                      range(population_size)]

    best_solution = None

    # Evolution process
    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations} start")

        # Evaluate fitness for each individual in the population
        if operation == 'dataset':
            fitness_scores = [
                (techniques, models, *fitness(dataset.copy(), techniques, models, protected_attribute, target_column))
                for (techniques, models) in population
            ]
        else:
            model, output_dir, X, y, X_df = model_info

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            fitness_scores = [
                (
                techniques, *fitness_model_optimization(model, techniques, X, y, protected_attribute, X_df, output_dir))
                for techniques in population
            ]

        # Print fitness scores and metrics for each individual
        for ind in fitness_scores:
            if operation == 'dataset':
                print(f"Techniques: {ind[0]}, Models: {ind[1]}")
            else:
                print(f"Techniques: {ind[0]}")
            print(f"Fitness Value: {ind[2]}")
            print(f"Performance and Fairness Metrics: {ind[3]}")
            print(f"Individual: {ind[:2] if operation == 'dataset' else ind[0]}\n")

        # Sort population by fitness value
        sorted_population = sorted(fitness_scores, key=lambda x: x[2])
        best_individuals = sorted_population[:population_size // 2]

        # Generate the next generation
        if operation == 'dataset':
            # Get the best techniques and models
            best_techniques = [technique for ind in best_individuals for technique in ind[0]]
            best_models = [model for ind in best_individuals for model in ind[1]]

            next_generation = []
            while len(next_generation) < population_size:
                parent_techniques = np.random.choice(best_techniques, 8, replace=False)
                parent_models = np.random.choice(best_models, 3, replace=False)
                child_techniques = np.random.choice(parent_techniques, 8, replace=False)
                child_models = np.random.choice(parent_models, 3, replace=False)

                # Mutate with a small probability
                if np.random.rand() < 0.1:
                    child_techniques = np.random.choice(techniques, size=8, replace=False)
                if np.random.rand() < 0.1:
                    child_models = np.random.choice(models, size=3, replace=False)

                next_generation.append((child_techniques, child_models))
            population = next_generation
        else:
            # Get the best techniques
            best_techniques = [technique for ind in best_individuals for technique in ind[0]]

            next_generation = []
            while len(next_generation) < population_size:
                parent_techniques = np.random.choice(best_techniques, 3, replace=False)
                child_techniques = np.random.choice(parent_techniques, 3, replace=False)

                # Mutate with a small probability
                if np.random.rand() < 0.1:
                    child_techniques = np.random.choice(model_optimization_techniques, size=3, replace=False)

                next_generation.append(child_techniques)
            population = next_generation

        print(f"End of generation {generation + 1}")

    # Get the best solution
    best_solution = min(fitness_scores, key=lambda x: x[2])
    if operation == 'dataset':
        print(f"Best solution: Techniques={best_solution[0]}, Models={best_solution[1]}, Fitness={best_solution[2]}")
        print("Performance and fairness metrics for the best solution:", best_solution[3])
        return best_solution
    else:
        print(f"Best solution: Techniques={best_solution[0]}, Fitness={best_solution[1]}")
        print("Performance and fairness metrics for the best solution:", best_solution[2])
        return best_solution
