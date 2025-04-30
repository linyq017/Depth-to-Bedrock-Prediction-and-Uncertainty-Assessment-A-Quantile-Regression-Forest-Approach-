import optuna
from sklearn.model_selection import KFold, cross_val_score
from quantile_forest import RandomForestQuantileRegressor
from tqdm import tqdm # Import tqdm
import os
import pandas as pd
def objective(trial, X, y):
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial (optuna.Trial): The current Optuna trial object.
        X (pd.DataFrame): Training features.
        y (pd.Series): Training target.

    Returns:
        float: The negative root mean squared error (RMSE) from cross-validation.
               Optuna minimizes this value.
    """
    # print(f"Starting trial {trial.number}...") # Remove this line to keep tqdm clean
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'oob_score': False,
        'n_jobs': -1 # Use all available cores for training each model
    }

    model = RandomForestQuantileRegressor(**params)
    # Use a fixed random_state for KFold for reproducibility of CV splits
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Use a high number of jobs for parallel cross-validation if resources allow
    # Note: n_jobs in cross_val_score is for parallelizing folds, not model training
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)

    rmse = -cv_scores.mean()
    # print(f"Trial {trial.number} completed with RMSE: {rmse:.3f}") # Remove this line to keep tqdm clean
    return rmse

def run_optimization(X_train_sub, y_train_sub, n_trials=20, output_path=None, frac=None):
    """
    Runs Optuna hyperparameter optimization study.

    Args:
        X_train_sub (pd.DataFrame): Subsampled training features.
        y_train_sub (pd.Series): Subsampled training target.
        n_trials (int, optional): The number of optimization trials. Defaults to 20.
        output_path (str, optional): Path to save the study results CSV. Defaults to None.
        frac (float, optional): The fraction value for naming the output file. Defaults to None.

    Returns:
        optuna.Study: The completed Optuna study object.
    """
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction='minimize')

    # Wrap the optimization loop with tqdm for a progress bar
    for _ in tqdm(range(n_trials), desc="Optuna Trials"):
         study.optimize(lambda trial: objective(trial, X_train_sub, y_train_sub), n_trials=1, show_progress_bar=False)


    print(f"Optimization completed. Best RMSE: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")

    if output_path:
        study_df = study.trials_dataframe()
        file_name = f'optuna_study_results_{frac}.csv' if frac is not None else 'optuna_study_results.csv'
        study_df.to_csv(os.path.join(output_path, file_name), index=False)
        print(f"Optuna study results saved to {os.path.join(output_path, file_name)}.")

    return study

if __name__ == '__main__':
    # Example usage:
    # Create some dummy data
    import numpy as np
    X_dummy = pd.DataFrame(np.random.rand(100, 10))
    y_dummy = pd.Series(np.random.rand(100))

    # Run a small optimization
    study = run_optimization(X_dummy, y_dummy, n_trials=5)
    print("Best params from example run:", study.best_params)