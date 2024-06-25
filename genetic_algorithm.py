import numpy as np
from fitness import fitness, fitness_model_optimization
from data_preparation import prepare_data_for_model_optimization
import os
import joblib

def genetic_algorithm(operation, dataset, protected_attribute, target_column, model_info=None, generations=10, population_size=10):
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

    if operation == 'dataset':
        population = [(np.random.choice(techniques), np.random.choice(models)) for _ in range(population_size)]
    else:
        population = [np.random.choice(model_optimization_techniques) for _ in range(population_size)]

    best_solution = None

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations} start")

        if operation == 'dataset':
            fitness_scores = [
                (technique, model, *fitness(dataset.copy(), technique, model, protected_attribute, target_column))
                for (technique, model) in population
            ]
        else:
            model, output_dir, X, y, X_df = model_info

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            fitness_scores = [
                (technique, *fitness_model_optimization(model, [technique], X, y, protected_attribute, X_df, output_dir))
                for technique in population
            ]

        for ind in fitness_scores:
            if operation == 'dataset':
                print(f"Technique: {ind[0]}, Model: {ind[1]}")
            else:
                print(f"Technique: {ind[0]}")
            print(f"Fitness Value: {ind[2]}")
            print(f"Performance and Fairness Metrics: {ind[3]}")
            print(f"Individual: {ind[:2] if operation == 'dataset' else ind[0]}\n")

        sorted_population = sorted(fitness_scores, key=lambda x: x[2])
        best_individuals = sorted_population[:population_size // 2]

        if operation == 'dataset':
            best_techniques = [ind[0] for ind in best_individuals]
            best_models = [ind[1] for ind in best_individuals]

            next_generation = []
            while len(next_generation) < population_size:
                parent_techniques = np.random.choice(best_techniques, 2, replace=False)
                parent_models = np.random.choice(best_models, 2, replace=False)
                child_technique = np.random.choice(parent_techniques)
                child_model = np.random.choice(parent_models)

                if np.random.rand() < 0.1:
                    child_technique = np.random.choice(techniques)
                if np.random.rand() < 0.1:
                    child_model = np.random.choice(models)

                next_generation.append((child_technique, child_model))
            population = next_generation
        else:
            best_techniques = [ind[0] for ind in best_individuals]

            next_generation = []
            while len(next_generation) < population_size:
                parent_techniques = np.random.choice(best_techniques, 2, replace=False)
                child_technique = np.random.choice(parent_techniques)

                if np.random.rand() < 0.1:
                    child_technique = np.random.choice(model_optimization_techniques)

                next_generation.append(child_technique)
            population = next_generation

        print(f"End of generation {generation + 1}")

    best_solution = min(fitness_scores, key=lambda x: x[2])
    if operation == 'dataset':
        print(f"Best solution: Technique={best_solution[0]}, Model={best_solution[1]}, Fitness={best_solution[2]}")
        print("Performance and fairness metrics for the best solution:", best_solution[3])
        return best_solution
    else:
        print(f"Best solution: Technique={best_solution[0]}, Fitness={best_solution[1]}")
        print("Performance and fairness metrics for the best solution:", best_solution[2])
        return best_solution
