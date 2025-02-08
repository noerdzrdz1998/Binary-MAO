import numpy as np
from typing import Callable, Tuple, Optional
from sklearn.base import ClassifierMixin
import warnings

def initialize_binary_population(pop_size: int, 
                                 dim: int) -> np.ndarray:
    """
    Initialize a binary population for the MAO algorithm.

    Parameters:
    -----------
    pop_size : int
        Number of individuals in the population.
    dim : int
        Number of dimensions (features).

    Returns:
    --------
    np.ndarray
        Initialized binary population with shape (pop_size, dim).
        Ensures no individual has all zeros.
    """
    population = np.random.randint(0, 2, (pop_size, dim))
    while np.any(np.sum(population, axis=1) == 0):
        zero_sum_indices = np.where(np.sum(population, axis=1) == 0)[0]
        for idx in zero_sum_indices:
            population[idx][np.random.randint(dim)] = 1
    return population

def divide_population(population: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide the population into male and female subgroups.

    Parameters:
    -----------
    population : np.ndarray
        The binary population.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Male and female populations.
    """
    half_size = len(population) // 2
    male_population = population[:half_size]
    female_population = population[half_size:]
    return male_population, female_population

def evaluate_binary_population(
    population: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model: Optional[Callable[[], ClassifierMixin]],
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    number_of_steps: int = 100
) -> np.ndarray:
    """
    Evaluate the fitness of the binary population.

    This function evaluates each individual in the population based on either a provided
    scikit-learn metric, the 'alpha' objective function, or the 'alpha-mean' objective function.

    Parameters:
    -----------
    population : np.ndarray
        The binary population to evaluate.
    X_train : np.ndarray
        Training dataset features.
    y_train : np.ndarray
        Training dataset labels.
    model : Optional[Callable[[], ClassifierMixin]]
        A callable that returns a scikit-learn compatible classifier.
        Required if `metric` is a scikit-learn metric.
    metric : Optional[Callable]
        Metric function to evaluate performance. If:
          - A scikit-learn metric is provided, it evaluates the model's performance.
          - 'alpha' is passed, the second objective function is used.
          - 'alpha-mean' is passed, the third objective function is used.
    number_of_steps : int, optional, default=100
        Number of steps for the alpha-based objective functions.

    Returns:
    --------
    np.ndarray
        Fitness values for each individual in the population.
    """
    fitness = []

    for individual in population:
        if np.sum(individual) == 0:
            fitness.append(-np.inf)
        else:
            selected_features = np.where(individual == 1)[0]
            X_train_selected = X_train[:, selected_features]

            if callable(metric):
                try:
                    model_instance = model()
                    model_instance.fit(X_train_selected, y_train)
                    y_pred = model_instance.predict(X_train_selected)
                    fitness.append(metric(y_train, y_pred))
                except Exception as e:
                    warnings.warn(f"Fitness evaluation error: {e}")
                    fitness.append(-np.inf)

            elif metric == 'alpha' or metric == 'alpha-mean':
                loo_mean = X_train_selected.mean(axis=1)
                min_mean, max_mean = loo_mean.min(), loo_mean.max()
                step = (max_mean - min_mean) / number_of_steps
                I_values = np.arange(min_mean + step, max_mean, step)
                vmax = 0
                for I in I_values:
                    below_I = loo_mean < I
                    above_I = loo_mean > I
                    v = np.sum((y_train == 0) & below_I) + np.sum((y_train == 1) & above_I)
                    w = np.sum((y_train == 0) & above_I) + np.sum((y_train == 1) & below_I)
                    if max(v, w) > vmax:
                        vmax = max(v, w)
                fitness.append(vmax)
                
            else:
                raise ValueError(f"Invalid metric: {metric}. Must be a callable, 'alpha', or 'alpha-mean'.")
    return np.array(fitness)

def transition_larvae_to_adult_binary(population: np.ndarray, 
                                      fitness: np.ndarray, 
                                      transition_prob: float, 
                                      lambda_factor: float) -> np.ndarray:
    """
    Perform the larvae-to-adult transition phase in binary MAO.

    Parameters:
    -----------
    population : np.ndarray
        Current binary population.
    fitness : np.ndarray
        Fitness values of the population.
    transition_prob : float
        Transition probability threshold.
    lambda_factor : float
        Factor for transitioning towards the best individual.

    Returns:
    --------
    np.ndarray
        Updated binary population.
    """
    best_index = np.argmax(fitness)
    best_individual = population[best_index]
    total_fitness = np.sum(fitness)
    for i in range(len(population)):
        individual_transition_prob = fitness[i] / total_fitness
        if individual_transition_prob < transition_prob:
            for j in range(len(population[i])):
                if np.random.rand() < lambda_factor:
                    population[i][j] = best_individual[j]
        else:
            population[i] = np.random.randint(0, 2, population.shape[1])
        if np.sum(population[i]) == 0:
            population[i][np.random.randint(len(population[i]))] = 1
    return population

def injury_and_restoration_binary(population: np.ndarray, 
                                  injury_prob: float, 
                                  regeneration_prob: float) -> np.ndarray:
    """
    Perform the injury and restoration phase in binary MAO.

    Parameters:
    -----------
    population : np.ndarray
        Current binary population.
    injury_prob : float
        Probability of injury for each individual.
    regeneration_prob : float
        Probability of regenerating an injured component.

    Returns:
    --------
    np.ndarray
        Updated binary population.
    """
    for i in range(len(population)):
        if np.random.rand() < injury_prob:
            injured_parts = np.random.rand(population.shape[1]) < regeneration_prob
            new_bits = np.random.randint(0, 2, np.sum(injured_parts))
            population[i][injured_parts] = new_bits
            if np.sum(population[i]) == 0:
                population[i][np.random.randint(len(population[i]))] = 1
    return population

def tournament_selection(population: np.ndarray, 
                         fitness: np.ndarray, 
                         k: int) -> np.ndarray:
    """
    Perform tournament selection to choose an individual based on fitness.

    Parameters:
    -----------
    population : np.ndarray
        Current binary population.
    fitness : np.ndarray
        Fitness values of the population.
    k : int
        Number of individuals participating in the tournament.

    Returns:
    --------
    np.ndarray
        Selected individual from the tournament.
    """
    selected_indices = np.random.choice(len(population), k, replace=False)
    selected_fitness = fitness[selected_indices]
    best_index = selected_indices[np.argmax(selected_fitness)]
    return population[best_index]

def reproduction_and_assortment_binary(
    male_population: np.ndarray,
    female_population: np.ndarray,
    male_fitness: np.ndarray,
    female_fitness: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model: Optional[Callable[[], ClassifierMixin]],
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    k: int = 3,
    number_of_steps: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform the reproduction and assortment phase in binary MAO using tournament selection.

    Parameters:
    -----------
    male_population : np.ndarray
        Male population.
    female_population : np.ndarray
        Female population.
    male_fitness : np.ndarray
        Fitness values of the male population.
    female_fitness : np.ndarray
        Fitness values of the female population.
    X_train : np.ndarray
        Training dataset features.
    y_train : np.ndarray
        Training dataset labels.
    model : Optional[Callable[[], ClassifierMixin]]
        A callable that returns an instance of a scikit-learn compatible classifier (e.g., SVM, k-NN).
    metric : Optional[Callable]
        Metric function to evaluate performance. If:
          - A scikit-learn metric is provided, it evaluates the model's performance.
          - 'alpha' is passed, the second objective function is used.
          - 'alpha-mean' is passed, the third objective function is used.
    k : int, optional, default=3
        Number of individuals participating in the tournament.
    number_of_steps : int, optional, default=10
        Number of steps for the alpha-based objective functions.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Updated male population, female population, and their fitness values.
    """
    new_population = []

    # Handle special case: individual dimension is 1
    if male_population.shape[1] == 1:
        return male_population, female_population, male_fitness, female_fitness
        
    # Regular case: dimension > 1
    for female in female_population:
        male = tournament_selection(male_population, male_fitness, k)
        offspring1, offspring2 = np.empty_like(male), np.empty_like(male)
        for i in range(len(male)):
            if np.random.rand() < 0.5:
                offspring1[i], offspring2[i] = male[i], female[i]
            else:
                offspring1[i], offspring2[i] = female[i], male[i]
        if np.sum(offspring1) == 0:
            offspring1[np.random.randint(len(offspring1))] = 1
        if np.sum(offspring2) == 0:
            offspring2[np.random.randint(len(offspring2))] = 1
        new_population.extend([offspring1, offspring2])

    new_population = np.array(new_population)
    new_fitness = evaluate_binary_population(new_population, X_train, y_train, model, metric, number_of_steps)
    combined_population = np.vstack((male_population, female_population, new_population))
    combined_fitness = np.hstack((male_fitness, female_fitness, new_fitness))
    indices = np.argsort(combined_fitness)[::-1]
    combined_population = combined_population[indices]
    combined_fitness = combined_fitness[indices]

    half_size = len(female_population)
    return (
        combined_population[:half_size],
        combined_population[half_size:2 * half_size],
        combined_fitness[:half_size],
        combined_fitness[half_size:2 * half_size],
    )

def MAO_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dim: int,
    model: Optional[Callable[[], ClassifierMixin]],
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    pop_size: int = 50,
    max_iter: int = 1000,
    early_stopping_steps: int = 5,
    transition_prob: float = 0.5,
    injury_prob: float = 0.3,
    regeneration_prob: float = 0.1,
    lambda_factor: float = 0.5,
    k: int = 3,
    number_of_steps: int = 100,
) -> np.ndarray:
    """
    Perform the binary MAO algorithm with feature selection. Includes early stopping based on a lack of improvement
    in the best fitness value over a specified number of generations.

    Parameters:
    -----------
    X_train : np.ndarray
        Training dataset features.
    y_train : np.ndarray
        Training dataset labels.
    dim : int
        Number of features (dimensions).
    model : Optional[Callable[[], ClassifierMixin]]
        A callable that returns a scikit-learn compatible classifier.
    metric : Optional[Callable]
        Metric function to evaluate performance. If:
          - A scikit-learn metric is provided, it evaluates the model's performance.
          - 'alpha' is passed, the second objective function is used.
          - 'alpha-mean' is passed, the third objective function is used.
    pop_size : int, optional, default=50
        Population size.
    max_iter : int, optional, default=1000
        Maximum number of iterations.
    early_stopping_steps : int, optional, default=5
        Number of generations with no improvement before stopping early.
    transition_prob : float, optional, default=0.5
        Transition probability for larvae-to-adult phase.
    injury_prob : float, optional, default=0.3
        Probability of injury.
    regeneration_prob : float, optional, default=0.1
        Probability of regeneration during injury phase.
    lambda_factor : float, optional, default=0.5
        Transition factor for larvae-to-adult phase.
    k : int, optional, default=3
        Number of individuals participating in tournament selection.
    number_of_steps : int, optional, default=10
        Number of steps for the alpha-based objective functions.

    Returns:
    --------
    np.ndarray
        Binary solution representing the selected features.
    """
    population = initialize_binary_population(pop_size, dim)
    male_population, female_population = divide_population(population)
    male_fitness = evaluate_binary_population(male_population, X_train, y_train, model, metric, number_of_steps)
    female_fitness = evaluate_binary_population(female_population, X_train, y_train, model, metric, number_of_steps)

    # Track the best global solution
    best_fitness = max(np.max(male_fitness), np.max(female_fitness))
    global_best_solution = (
        male_population[np.argmax(male_fitness)]
        if np.max(male_fitness) > np.max(female_fitness)
        else female_population[np.argmax(female_fitness)]
    )

    # Main loop with early stopping
    no_improvement_counter = 0
    for _ in range(max_iter):
        male_population = transition_larvae_to_adult_binary(male_population, male_fitness, transition_prob, lambda_factor)
        female_population = transition_larvae_to_adult_binary(female_population, female_fitness, transition_prob, lambda_factor)
        male_fitness = evaluate_binary_population(male_population, X_train, y_train, model, metric, number_of_steps)
        female_fitness = evaluate_binary_population(female_population, X_train, y_train, model, metric, number_of_steps)
        male_population = injury_and_restoration_binary(male_population, injury_prob, regeneration_prob)
        female_population = injury_and_restoration_binary(female_population, injury_prob, regeneration_prob)
        male_fitness = evaluate_binary_population(male_population, X_train, y_train, model, metric, number_of_steps)
        female_fitness = evaluate_binary_population(female_population, X_train, y_train, model, metric, number_of_steps)
        male_population, female_population, male_fitness, female_fitness = reproduction_and_assortment_binary(
            male_population, female_population, male_fitness, female_fitness, X_train, y_train, model, metric, k, number_of_steps
        )
        # Update the global best solution
        current_best_fitness = max(np.max(male_fitness), np.max(female_fitness))
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            global_best_solution = (
                male_population[np.argmax(male_fitness)]
                if np.max(male_fitness) > np.max(female_fitness)
                else female_population[np.argmax(female_fitness)]
            )
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        # Early stopping condition
        if no_improvement_counter >= early_stopping_steps:
            break

    return global_best_solution

