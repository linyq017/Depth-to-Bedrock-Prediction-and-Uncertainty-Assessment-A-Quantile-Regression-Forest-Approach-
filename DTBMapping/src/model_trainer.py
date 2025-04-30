import joblib
import os
from quantile_forest import RandomForestQuantileRegressor
import pandas as pd # Import pandas for type hinting

def train_and_save_model(X_train: pd.DataFrame, y_train: pd.Series, best_params: dict, output_dir: str, frac: float):
    """
    Trains a RandomForestQuantileRegressor model and saves it.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        best_params (dict): Dictionary of best hyperparameters from optimization.
        output_dir (str): Directory to save the trained model.
        frac (float): The subsample fraction (used in filename).

    Returns:
        RandomForestQuantileRegressor: The trained model.
    """
    print("Training model with best parameters...")
    qrf = RandomForestQuantileRegressor(**best_params, random_state=42) # Added random_state for reproducibility
    qrf.fit(X_train, y_train)

    model_path = os.path.join(output_dir, f'qrf_mode_{frac}l.pkl')
    joblib.dump(qrf, model_path)
    print(f"Model trained and saved to {model_path}.")
    return qrf

if __name__ == '__main__':
    # Example usage:
    # Create some dummy data
    import numpy as np
    X_dummy = pd.DataFrame(np.random.rand(100, 10))
    y_dummy = pd.Series(np.random.rand(100))
    best_params_dummy = {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 5}
    output_dir_dummy = './model_tests'
    os.makedirs(output_dir_dummy, exist_ok=True)

    trained_model = train_and_save_model(X_dummy, y_dummy, best_params_dummy, output_dir_dummy, frac=1.0)
    print("Example model trained.")
    # Clean up dummy directory
    # os.remove(os.path.join(output_dir_dummy, 'qrf_mode_1.0l.pkl'))
    # os.rmdir(output_dir_dummy)