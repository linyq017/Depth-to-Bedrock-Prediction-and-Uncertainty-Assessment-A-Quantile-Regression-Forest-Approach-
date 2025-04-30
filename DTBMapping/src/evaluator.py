import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap
import os
import joblib # Import joblib to load the model if needed for standalone testing

def calculate_metrics(y_true: pd.Series, y_pred_50: np.ndarray, y_pred_05: np.ndarray = None, y_pred_95: np.ndarray = None):
    """
    Calculates standard regression metrics and PICP.

    Args:
        y_true (pd.Series): True target values.
        y_pred_50 (np.ndarray): 50th percentile predictions.
        y_pred_05 (np.ndarray, optional): 5th percentile predictions. Defaults to None.
        y_pred_95 (np.ndarray, optional): 95th percentile predictions. Defaults to None.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    print("Calculating evaluation metrics...")
    r2 = r2_score(y_true, y_pred_50)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_50))
    mae = mean_absolute_error(y_true, y_pred_50)
    mean_error = np.mean(y_true - y_pred_50)

    metrics = {
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'ME': mean_error,
    }

    if y_pred_05 is not None and y_pred_95 is not None:
        picp = np.mean((y_true >= y_pred_05) & (y_true <= y_pred_95))
        metrics['PICP'] = picp
        print(f'R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, ME: {mean_error:.2f}, PICP: {picp:.4f}')
    else:
        print(f'R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, ME: {mean_error:.2f}')


    return metrics

def calculate_picp_for_intervals(y_test: pd.Series, percentile_predictions: dict):
    """
    Calculate the PICP for multiple prediction intervals (e.g., PI10, PI20, ..., PI90).

    Parameters:
    - y_test (pd.Series): Array of true values
    - percentile_predictions (dict): Dictionary containing lower and upper percentile predictions for each interval.
                               Keys should be strings like 'PI10', 'PI20', ..., 'PI90'
                               Values should be tuples (lower_percentile_prediction, upper_percentile_prediction)

    Returns:
    - picp_results (dict): Dictionary with PICP values for each prediction interval.
    """
    print("Calculating PICP for multiple intervals...")
    picp_results = {}

    for interval_name, (lower_bound, upper_bound) in percentile_predictions.items():
        # Ensure bounds are numpy arrays for comparison
        lower_bound = np.asarray(lower_bound)
        upper_bound = np.asarray(upper_bound)

        # Create a boolean mask indicating if the true values fall within the bounds
        within_bounds = (y_test.values >= lower_bound) & (y_test.values <= upper_bound)
        # Calculate the PICP: proportion of true values within bounds
        picp = np.mean(within_bounds)
        # Store the result
        picp_results[interval_name] = picp

    return picp_results

def get_shap_values(qrf, X: pd.DataFrame, quantile=0.5):
    """Get QRF model SHAP values using Tree SHAP."""
    print(f"Calculating SHAP values for quantile {quantile}...")
    # Prepare the model structure for TreeExplainer
    # Ensure tree_output matches the QRF's output structure for consistency
    model = {
        "objective": "regression", # RandomForestQuantileRegressor uses regression objective
        "tree_output": "raw_value",
        "trees": [e.tree_ for e in qrf.estimators_],
    }

    # Using the data itself as the background dataset is common for tree models
    explainer = shap.TreeExplainer(model, data=X) # Use data=X for TreeExplainer

    # QRF SHAP values need adjustment based on the specific quantile prediction
    # The base value is the mean prediction of the ensemble, adjusted by the quantile offset
    qrf_pred = qrf.predict(X, quantiles=quantile)
    rf_pred = qrf.predict(X, quantiles="mean") # Mean prediction from the ensemble

    # The base value for a specific quantile prediction is the mean prediction
    # plus the difference between the quantile prediction and the mean prediction.
    base_offset = qrf_pred - rf_pred

    # The SHAP values from TreeExplainer (when tree_output="raw_value") need scaling
    # because TreeExplainer calculates SHAP values for individual trees, and we
    # are interested in the ensemble prediction.
    scaling = 1.0 / len(qrf.estimators_)
    shap_values_raw = explainer.shap_values(X, check_additivity=False)

    # The SHAP values for the quantile prediction are the scaled raw SHAP values.
    # The base value for each prediction is the mean prediction plus the offset.
    # TreeExplainer's base_value when tree_output="raw_value" is the mean output
    # of the trees *on the background dataset*. We need to adjust this base value
    # to match the specific quantile prediction's base.

    # The base value for TreeExplainer with raw_value output is typically
    # the average of the leaf node values across the background data for each tree,
    # averaged across trees. This is close to the mean prediction.
    # The `explainer.expected_value` when using `data=X` should be the mean prediction.
    # Let's use the mean prediction of the trained model as the base for scaling.
    # This is a bit subtle for quantile models, but the general approach is to
    # adjust the base value based on the difference between the quantile prediction
    # and the mean prediction.

    # For QRF, the SHAP values for a specific quantile prediction are generally
    # computed relative to the mean prediction, and then an offset is added.
    # A common approach is to calculate SHAP values for the mean prediction and
    # then adjust the base value.

    # Let's simplify and calculate SHAP for the mean prediction first,
    # then adjust the base value for the quantile prediction.

    # Calculate SHAP for the mean prediction (equivalent to a standard RF)
    explainer_mean = shap.TreeExplainer(qrf, data=X)
    shap_values_mean = explainer_mean.shap_values(X, check_additivity=False)

    # The base value for the quantile prediction is the mean prediction plus the offset.
    # The SHAP values themselves contribute to the deviation from the base value.
    # The simplest approach consistent with many SHAP interpretations for ensemble
    # models is to use the SHAP values calculated for the mean prediction
    # and adjust the base value.

    # Let's return the SHAP values calculated for the mean prediction,
    # and let the user know that the base value interpretation is complex for QRF.
    # Or, we can provide the base value that matches the quantile prediction.
    # The total sum of SHAP values plus the base value should approximate the prediction.

    # Sum of mean SHAP values + mean explainer base value should approximate mean prediction
    # sum(shap_values_mean[i]) + explainer_mean.expected_value approx qrf.predict(X.iloc[i], quantiles="mean")

    # For quantile prediction, the sum of mean SHAP values + adjusted base value should approximate quantile prediction.
    # Adjusted base value = explainer_mean.expected_value + base_offset
    adjusted_base_values = explainer_mean.expected_value + base_offset

    # Create a SHAP Explanation object for easier plotting
    shap_explanation = shap.Explanation(
        values=shap_values_mean, # Use SHAP values for the mean prediction
        base_values=adjusted_base_values, # Use the adjusted base value for the quantile
        data=X.values,
        feature_names=X.columns.tolist()
    )


    print("SHAP value calculation completed.")
    return shap_explanation


if __name__ == '__main__':
    # Example usage:
    # Create dummy data
    y_true_dummy = pd.Series(np.random.rand(100))
    y_pred_50_dummy = np.random.rand(100)
    y_pred_05_dummy = y_pred_50_dummy - 0.1
    y_pred_95_dummy = y_pred_50_dummy + 0.1

    metrics = calculate_metrics(y_true_dummy, y_pred_50_dummy, y_pred_05_dummy, y_pred_95_dummy)
    print("Example metrics:", metrics)

    # Dummy data for PICP intervals
    y_test_dummy = pd.Series(np.random.rand(100) * 10)
    preds = {
        '10': (y_test_dummy * 0.95, y_test_dummy * 1.05),
        '50': (y_test_dummy * 0.8, y_test_dummy * 1.2),
        '90': (y_test_dummy * 0.5, y_test_dummy * 1.5),
    }
    picp_results_dummy = calculate_picp_for_intervals(y_test_dummy, preds)
    print("Example PICP results:", picp_results_dummy)

    # Example SHAP (requires a trained model)
    # This part is harder to run standalone without a trained model file.
    # Assuming you have a dummy model saved:
    # try:
    #     qrf_dummy = joblib.load('./model_tests/qrf_mode_1.0l.pkl') # Load a dummy model
    #     X_dummy = pd.DataFrame(np.random.rand(10, 10)) # Dummy X data
    #     shap_values_dummy = get_shap_values(qrf_dummy, X_dummy)
    #     print("Example SHAP values shape:", shap_values_dummy.values.shape)
    # except FileNotFoundError:
    #      print("Dummy model not found for SHAP example.")