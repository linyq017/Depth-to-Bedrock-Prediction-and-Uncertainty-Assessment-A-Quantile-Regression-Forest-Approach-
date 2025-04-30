import matplotlib.pyplot as plt
import numpy as np
import os
import shap
import pandas as pd # Import pandas for type hinting
from sklearn.metrics import mean_squared_error

def plot_pred_vs_obs(y_pred: np.ndarray, y_true: pd.Series, rmse: float, mean_error: float, sub_folder: str, frac: float, train_sample_size: int, test_sample_size: int):
    """
    Generates and saves a predicted vs observed plot.

    Args:
        y_pred (np.ndarray): Predicted values (specifically the 50th percentile).
        y_true (pd.Series): True observed values.
        rmse (float): Root Mean Squared Error.
        mean_error (float): Mean Error.
        sub_folder (str): Directory to save the plot.
        frac (float): Subsample fraction (used in filename).
        train_sample_size (int): Number of training samples.
        test_sample_size (int): Number of testing samples.
    """
    print("Creating predicted vs observed plot...")
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams['font.size'] = 12

    # Diagonal line (perfect fit)
    min_val = min(y_pred.min(), y_true.min())
    max_val = max(y_pred.max(), y_true.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='grey', linestyle='--', linewidth=2, alpha = 0.5)

    # Annotate sample sizes and metrics
    # Position annotations relative to max_val for better scaling
    text_x_pos = 0.15 * max_val
    text_y_pos_start = 0.90 * max_val
    line_spacing = 0.04 * max_val

    ax.text(text_x_pos, text_y_pos_start, f'Train: n = {train_sample_size}', fontsize=12)
    ax.text(text_x_pos, text_y_pos_start - line_spacing, f'Test: n = {test_sample_size}', fontsize=12)
    ax.text(text_x_pos, text_y_pos_start - 2 * line_spacing, f'RMSE: {rmse:.2f}', fontsize=12)
    ax.text(text_x_pos, text_y_pos_start - 3 * line_spacing, f'ME: {mean_error:.2f}', fontsize=12)

    # Hexbin plot
    hb = ax.hexbin(y_true, y_pred, gridsize=50, mincnt=1, bins='log', cmap='viridis', alpha=0.8)
    ax.grid(True, linestyle='--', color='grey', alpha=0.5, linewidth=0.5)

    # Set axis limits to start at 0
    # Add a small buffer to max_val to prevent points from being cut off at the edge
    buffer = (max_val - min_val) * 0.05
    ax.set_xlim([0, max_val + buffer])
    ax.set_ylim([0, max_val + buffer])
    # Ensure min_val is not negative for setting limit
    ax.set_xlim([max(0, min_val), max_val + buffer])
    ax.set_ylim([max(0, min_val), max_val + buffer])


    # Add a colorbar inside the plot space
    # Adjust colorbar position based on plot size
    cax = fig.add_axes([0.78, 0.35, 0.03, 0.3])  # [left, bottom, width, height] - relative coordinates
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label('log(Count)')

    ax.set_ylabel('Predicted Values')
    ax.set_xlabel('Observed Values')
    ax.set_title(f'Predicted vs Observed (Subsample Fraction: {frac})') # Add title

    # Save plot
    plt.savefig(os.path.join(sub_folder, f"predicted_vs_observe_{frac}.png"), dpi=600)
    plt.close(fig)
    print("Predicted vs observed plot saved.")

def plot_shap_summary(shap_values: shap.Explanation, sub_folder: str, frac: float):
    """
    Generates and saves a SHAP summary plot.

    Args:
        shap_values (shap.Explanation): SHAP values as a shap.Explanation object.
        sub_folder (str): Directory to save the plot.
        frac (float): Subsample fraction (used in filename).
    """
    print("Creating SHAP summary plot...")
    plt.figure(figsize=(10, 8)) # Adjust figure size if needed
    shap.summary_plot(shap_values, show=False)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.savefig(os.path.join(sub_folder, f"SHAP_summary_{frac}.png"), dpi=600, bbox_inches='tight') # Use bbox_inches='tight'
    plt.close()
    print("SHAP summary plot saved.")

if __name__ == '__main__':
    # Example usage:
    # Create dummy data
    y_true_dummy = pd.Series(np.random.rand(100) * 10)
    y_pred_dummy = np.random.rand(100) * 10
    rmse_dummy = np.sqrt(mean_squared_error(y_true_dummy, y_pred_dummy))
    me_dummy = np.mean(y_true_dummy - y_pred_dummy)
    output_dir_dummy = './plot_tests'
    os.makedirs(output_dir_dummy, exist_ok=True)

    plot_pred_vs_obs(y_pred_dummy, y_true_dummy, rmse_dummy, me_dummy, output_dir_dummy, frac=0.5, train_sample_size=50, test_sample_size=100)
    print("Example predicted vs observed plot created.")

    # Example SHAP plot (requires dummy SHAP values and feature names)
    # Create dummy SHAP explanation
    X_dummy = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
    shap_values_array_dummy = np.random.rand(100, 5) - 0.5
    base_values_dummy = np.random.rand(100) * 5
    shap_explanation_dummy = shap.Explanation(
        values=shap_values_array_dummy,
        base_values=base_values_dummy,
        data=X_dummy.values,
        feature_names=X_dummy.columns.tolist()
    )
    plot_shap_summary(shap_explanation_dummy, output_dir_dummy, frac=0.5)
    print("Example SHAP plot created.")

    # Clean up dummy directory
    # import shutil
    # shutil.rmtree(output_dir_dummy)