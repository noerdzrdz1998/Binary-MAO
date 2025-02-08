import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, RepeatedKFold, train_test_split
from typing import Callable, List, Dict, Any, Optional
from mao_binary import MAO_binary

def model_with_metaheuristic_feature_selection(
    datasets: List[np.ndarray],
    datasets_names: List[str],
    model: Callable,
    mao_metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    evaluation_metric: Callable[[np.ndarray, np.ndarray], float] = None,
    validation_method: str = "stratified_kfold",
    validation_params: Dict[str, Any] = {"n_splits": 5, "shuffle": True, "random_state": 42},
    pop_size: int = 50,
    max_iter: int = 1000,
    early_stopping_steps: int = 10,
    transition_prob: float = 0.5,
    injury_prob: float = 0.3,
    regeneration_prob: float = 0.1,
    lambda_factor: float = 0.5,
    k: int = 3,
    number_of_steps: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """
    Perform feature selection using a metaheuristic algorithm and evaluate the performance of a given ML model.

    Parameters:
    -----------
    datasets : List[np.ndarray]
        A list of datasets, where each dataset is a Pandas DataFrame with features and a binary class column.
    datasets_names : List[str]
        Names of the datasets.
    model : Callable
        A callable that returns a scikit-learn compatible classifier instance with desired hyperparameters.
    mao_metric : Optional[str]
        The metric used in the MAO_binary algorithm ('alpha', 'alpha-mean', or a scikit-learn compatible metric).
    evaluation_metric : Callable
        A scikit-learn metric function (e.g., F1-score, accuracy) used to evaluate the ML model after feature selection.
    validation_method : str, optional, default="stratified_kfold"
        Validation method to use: "holdout", "kfold", "stratified_kfold", "loocv", or "repeated_kfold".
    validation_params : Dict[str, Any], optional
        Parameters for the validation method, including `random_state` for reproducibility.
    pop_size : int, optional, default=50
        Population size for the metaheuristic algorithm.
    max_iter : int, optional, default=1000
        Maximum number of iterations for the metaheuristic algorithm.
    early_stopping_steps : int, optional, default=5
        Number of iterations with no improvement before stopping early.
    transition_prob : float, optional, default=0.5
        Transition probability for the larvae-to-adult phase.
    injury_prob : float, optional, default=0.3
        Probability of injury in the metaheuristic algorithm.
    regeneration_prob : float, optional, default=0.1
        Probability of regeneration during the injury phase.
    lambda_factor : float, optional, default=0.5
        Factor for transitioning towards the best individual.
    k : int, optional, default=3
        Number of individuals participating in tournament selection during reproduction.
    number_of_steps : int, optional, default=100
        Number of steps for the alpha-based objective functions.

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        A dictionary containing performance metrics and selected features for each dataset.

    Notes:
    ------
    - Allows any scikit-learn compatible classifier and performance metric for evaluation.
    - Supports custom metrics ('alpha' and 'alpha-mean') in the MAO_binary algorithm.
    - For 'alpha-mean', averages the selected features before passing them to the ML model for evaluation.
    - Supports multiple validation strategies: holdout, k-fold, stratified k-fold, leave-one-out, repeated k-fold.
    """
    results = {}

    for data, name in zip(datasets, datasets_names):
        dataset_start_time = time.time()

        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Define validation method
        if validation_method == "holdout":
            if "test_size" not in validation_params:
                validation_params["test_size"] = 0.3
            X_train, X_test, y_train, y_test = train_test_split(X, y, **validation_params)

            # Run MAO_binary for holdout
            best_subset = MAO_binary(
                X_train,
                y_train,
                X.shape[1],
                model=model,
                metric=mao_metric,
                pop_size=pop_size,
                max_iter=max_iter,
                early_stopping_steps=early_stopping_steps,
                transition_prob=transition_prob,
                injury_prob=injury_prob,
                regeneration_prob=regeneration_prob,
                lambda_factor=lambda_factor,
                k=k,
                number_of_steps=number_of_steps,
            )

            selected_features = np.where(best_subset == 1)[0]

            # Handle feature selection based on MAO metric
            if mao_metric == "alpha-mean":
                X_train_selected = X_train[:, selected_features].mean(axis=1).reshape(-1, 1)
                X_test_selected = X_test[:, selected_features].mean(axis=1).reshape(-1, 1)
            else:
                X_train_selected = X_train[:, selected_features]
                X_test_selected = X_test[:, selected_features]

            # Evaluate the model using the evaluation metric
            model_instance = model()
            model_instance.fit(X_train_selected, y_train)
            y_pred = model_instance.predict(X_test_selected)
            performance_score = evaluation_metric(y_test, y_pred)

            performance_scores = [performance_score]

        else:
            # Handle other validation methods
            if validation_method == "kfold":
                validation_instance = KFold(**validation_params)
            elif validation_method == "stratified_kfold":
                validation_instance = StratifiedKFold(**validation_params)
            elif validation_method == "loocv":
                validation_instance = LeaveOneOut()
            elif validation_method == "repeated_kfold":
                validation_instance = RepeatedKFold(**validation_params)
            else:
                raise ValueError(f"Unsupported validation method: {validation_method}")

            validation_splits = [
                (X[train_idx], X[test_idx], y[train_idx], y[test_idx])
                for train_idx, test_idx in validation_instance.split(X, y)
            ]

            performance_scores = []

            for X_train, X_test, y_train, y_test in validation_splits:
                best_subset = MAO_binary(
                    X_train,
                    y_train,
                    X.shape[1],
                    model=model,
                    metric=mao_metric,
                    pop_size=pop_size,
                    max_iter=max_iter,
                    early_stopping_steps=early_stopping_steps,
                    transition_prob=transition_prob,
                    injury_prob=injury_prob,
                    regeneration_prob=regeneration_prob,
                    lambda_factor=lambda_factor,
                    k=k,
                    number_of_steps=number_of_steps,
                )

                selected_features = np.where(best_subset == 1)[0]

                # Handle feature selection for alpha-mean
                if mao_metric == "alpha-mean":
                    X_train_selected = X_train[:, selected_features].mean(axis=1).reshape(-1, 1)
                    X_test_selected = X_test[:, selected_features].mean(axis=1).reshape(-1, 1)
                else:
                    X_train_selected = X_train[:, selected_features]
                    X_test_selected = X_test[:, selected_features]

                # Train and evaluate model
                model_instance = model()
                model_instance.fit(X_train_selected, y_train)
                y_pred = model_instance.predict(X_test_selected)
                performance_score = evaluation_metric(y_test, y_pred)

                performance_scores.append(performance_score)

        dataset_end_time = time.time()
        elapsed_time = dataset_end_time - dataset_start_time

        # Save results for the dataset
        results[name] = {
            "mean_metric": np.mean(performance_scores),
            "selected_features": selected_features.tolist(),
            "processing_time": elapsed_time,
        }

    return results

