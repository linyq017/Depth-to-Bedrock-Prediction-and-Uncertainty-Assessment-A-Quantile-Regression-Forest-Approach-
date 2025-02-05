import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna.visualization as vis
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from datetime import datetime
import os
from tqdm import tqdm
import shap
import joblib


# Load and prepare training and testing datasets
soildepth_train = pd.read_csv('/workspace/data/soildepth/DepthData/csvs/training_faultline.csv',sep = ',', decimal= '.')
soildepth_test = pd.read_csv('/workspace/data/soildepth/DepthData/csvs/testing_faultline.csv',sep = ',', decimal= '.')
# Subset relevant columns
def subset_columns(data):
    print("Subsetting the relevant columns...")
    return data[['DJUP', 'N', 'E',  'Aspect50', 'ProCur50', 'RTP20_20', 'RTP50_50', 'Slope20', 'DEM', 'EAS1ha', 'DI2m', 'CVA', 'SDFS', 'DFME', 'Rugged', 
                  'HKDepth', 'MSRM', 'MED', 'jbas_merged_grus', 'jbas_merged_hall', 'jbas_merged_isalvssediment', 'jbas_merged_lera', 'jbas_merged_moran', 'jbas_merged_sand', 'jbas_merged_sjo', 'jbas_merged_torv', 
                  'karttyp_2','karttyp_3', 'karttyp_4', 'karttyp_5', 'karttyp_6', 'karttyp_7',
                    'karttyp_8', 'karttyp_9', 'tekt_n_0', 'tekt_n_67', 'tekt_n_68',
                    'tekt_n_69', 'tekt_n_70', 'tekt_n_72', 'tekt_n_79', 'tekt_n_82',
                    'tekt_n_88', 'tekt_n_337', 'tekt_n_346', 'tekt_n_368',
                    'tekt_n_380', 'tekt_n_387', 'tekt_n_388', 'tekt_n_389', 'tekt_n_390',
                    'tekt_n_394', 'tekt_n_1939', 'Geomorphon_Flat', 'Geomorphon_Footslope',
                    'Geomorphon_Hollow(concave)', 'Geomorphon_Peak(summit)',
                    'Geomorphon_Pit(depression)', 'Geomorphon_Ridge', 'Geomorphon_Shoulder',
                    'Geomorphon_Slope', 'Geomorphon_Spur(convex)', 'Geomorphon_Valley',
                    'DistanceToDeformation']]
 
# Objective function for Optuna optimization
def objective(trial, X, y):
    print(f"Starting trial {trial.number}...")
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15), 
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'oob_score': False,
        'n_jobs': -1
    }
    
    model = RandomForestQuantileRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=64)
    
    rmse = -cv_scores.mean()
    print(f"Trial {trial.number} completed with RMSE: {rmse:.3f}")
    return rmse

 
# Train and save model
def train_and_save_model(X_train, y_train, best_params, sub_folder):
    qrf = RandomForestQuantileRegressor(**best_params).fit(X_train, y_train)
    joblib.dump(qrf, os.path.join(sub_folder, f'qrf_mode_{frac}l.pkl'))
    print("Model trained and saved.")
    return qrf

# Calculate SHAP values
def get_shap_values(qrf, X, quantile=0.5):
    """Get QRF model SHAP values using Tree SHAP."""
    model = {
        "objective": qrf.criterion,
        "tree_output": "raw_value",
        "trees": [e.tree_ for e in qrf.estimators_],
    }

    explainer = shap.TreeExplainer(model, X)
    qrf_pred = qrf.predict(X, quantiles=quantile)
    rf_pred = qrf.predict(X, quantiles="mean")

    scaling = 1.0 / len(qrf.estimators_)
    base_offset = qrf_pred - rf_pred

    explainer.expected_value *= scaling
    explainer.expected_value = np.tile(explainer.expected_value, len(X)) + np.array(base_offset)

    shap_values = explainer(X, check_additivity=False)
    shap_values.base_values = np.diag(shap_values.base_values)

    return shap_values

def calculate_picp_for_intervals(y_test, percentile_predictions):
    """
    Calculate the PICP for multiple prediction intervals (PI10, PI20, ..., PI90).

    Parameters:
    - y_test: Array of true values
    - percentile_predictions: Dictionary containing lower and upper percentile predictions for each interval
                              Keys should be strings like 'PI10', 'PI20', ..., 'PI90'
                              Values should be tuples (lower_percentile_prediction, upper_percentile_prediction)
    
    Returns:
    - picp_results: Dictionary with PICP values for each prediction interval
    """
    picp_results = {}
    
    for interval, (lower_bound, upper_bound) in percentile_predictions.items():
        # Create a boolean mask indicating if the true values fall within the bounds
        within_bounds = (y_test >= lower_bound) & (y_test <= upper_bound)
        # Calculate the PICP: proportion of true values within bounds
        picp = np.mean(within_bounds)
        # Store the result
        picp_results[interval] = picp
    
    return picp_results


# Create output directory if it doesn't exist
def create_output_directory(output_folder):
    # Generate a timestamp to create a unique subfolder
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    sub_folder = os.path.join(output_folder, timestamp)  # Use output_folder instead
    os.makedirs(sub_folder, exist_ok=True)  # Create the subfolder
    print(f"Created subfolder at {sub_folder}.")  # Print the correct path
    return sub_folder  # Return the path of the subfolder

# Generate prediction vs observed plot
def plot_pred_vs_obs(y_pred_50, y_test, rmse, mean_error, sub_folder):
    print("Creating predicted vs observed plot...")
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.rcParams['font.size'] = 12  

    # Diagonal line (perfect fit)
    min_val = min(y_pred_50.min(), y_test.min())
    max_val = max(y_pred_50.max(), y_test.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='grey', linestyle='--', linewidth=2, alpha = 0.5)
    # Annotate total sample size
    plt.text(0.15* y_test.max(), 0.90 * y_test.max(), f'Train: n = {len(train_samples)}', fontsize=12)
    plt.text(0.15* y_test.max(), 0.86 * y_test.max(), f'Test: n = {len(y_test)}', fontsize=12)
    # R² and RMSE ME annotations
    plt.text(0.15* y_test.max(), 0.78 * y_test.max(), f'RMSE: {rmse:.2f}', fontsize=12)
    plt.text(0.15* y_test.max(), 0.74 * y_test.max(), f'ME: {mean_error:.2f}', fontsize=12)

    # Hexbin plot of predicted vs observed values with log scaling for color
    hb = ax.hexbin(y_test, y_pred_50, gridsize=50, mincnt=1, bins='log', cmap='viridis', alpha= 0.8) 
    ax.grid(True, linestyle='--', color='grey', alpha = 0.5, linewidth=0.5)
    # Set axis limits to start at 0
    ax.set_xlim([0, max_val])
    ax.set_ylim([0, max_val])

     # Add a colorbar inside the plot space
    cax = fig.add_axes([0.78, 0.35, 0.03, 0.3])  # [left, bottom, width, height]
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label('log(Count)')
    ax.set_ylabel('Predicted Values')
    ax.set_xlabel('Observed Values') 

    # Save plot
    plt.savefig(os.path.join(sub_folder,  f"predicted_vs_observe_{frac}.png"), dpi=600)
    plt.close(fig)
    print("Predicted vs observed plot saved.")

# Subsample the training data
def subsample_training_data(X_train, y_train, frac=1.0):
    print(f"Subsampling {frac * 100}% of the training data...")
    X_train_sub = X_train.sample(frac=frac)
    y_train_sub = y_train.loc[X_train_sub.index]  # Keep the corresponding y values
    return X_train_sub, y_train_sub

# Main script
if __name__ == '__main__':
    output_folder = '/workspace/data/soildepth/Hypertune'
    sub_folder = create_output_directory(output_folder)

    train_data = subset_columns(soildepth_train)
    test_data = subset_columns(soildepth_test)

    # Train/test split
    X_train = train_data.drop(columns='DJUP')
    y_train = train_data['DJUP']
    X_test = test_data.drop(columns='DJUP')
    y_test = test_data['DJUP']

    print(f'X_train: {len(X_train)}, y_train: {len(y_train)}, X_test: {len(X_test)}, y_test: {len(y_test)}')

    # Subsample fractions for training
    subsample_fractions = [ 0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]#1.0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,

    # List to store RMSE, R², MAE, ME, PICP, and fraction values
    results = []

    for frac in subsample_fractions:
        print(f"\nStarting subsampling with fraction {frac}")
        
        # Subsample the training data
        X_train_sub, y_train_sub = subsample_training_data(X_train, y_train, frac=frac)
        print(f'Subsampled X_train: {len(X_train_sub)}, y_train: {len(y_train_sub)}')
        print('Subsampled X_train columns :', X_train_sub.columns)
        # Total sample size
        test_samples = y_test
        train_samples = X_train_sub

        # Hyperparameter optimization
        study = optuna.create_study(direction='minimize')
        print("Starting hyperparameter optimization...")
        for _ in tqdm(range(20), desc="Trials"):
            study.optimize(lambda trial: objective(trial, X_train_sub, y_train_sub), n_trials=1)

        best_params = study.best_params
        print(f"Best hyperparameters: {best_params}")
        
        # Save the study results
        study_df = study.trials_dataframe()
        study_df.to_csv(os.path.join(sub_folder, f'optuna_study_results_{frac}.csv'), index=False)
        print(f"Optuna study results saved for fraction {frac}.")

        # Train the model and evaluate on the full test set
        print(f"Model training starts for {frac}.")
        qrf = train_and_save_model(X_train_sub, y_train_sub, best_params, sub_folder)
        y_pred = qrf.predict(X_test, quantiles=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        y_pred_05 = y_pred[:, 0]
        y_pred_10 = y_pred[:, 1]  
        y_pred_15 = y_pred[:, 2]
        y_pred_20 = y_pred[:, 3]  
        y_pred_25 = y_pred[:, 4]
        y_pred_30 = y_pred[:, 5]  
        y_pred_35 = y_pred[:, 6]
        y_pred_40 = y_pred[:, 7]  
        y_pred_45 = y_pred[:, 8]
        y_pred_50 = y_pred[:, 9]  
        y_pred_55 = y_pred[:, 10]
        y_pred_60 = y_pred[:, 11]  
        y_pred_65 = y_pred[:, 12]
        y_pred_70 = y_pred[:, 13]  
        y_pred_75 = y_pred[:, 14]
        y_pred_80 = y_pred[:, 15]  
        y_pred_85 = y_pred[:, 16]
        y_pred_90 = y_pred[:, 17]  
        y_pred_95 = y_pred[:, 18]
        print(f"prediction completed for {frac}.")

        percentile_predictions = {
            '10': (y_pred_45, y_pred_55),
            '20': (y_pred_40, y_pred_60),
            '30': (y_pred_35, y_pred_65),
            '40': (y_pred_30, y_pred_70),
            '50': (y_pred_25, y_pred_75),
            '60': (y_pred_20, y_pred_80),
            '70': (y_pred_15, y_pred_85),
            '80': (y_pred_10, y_pred_90),
            '90': (y_pred_05, y_pred_95)
        }

        # Calculate PICP for each interval
        picp_results = calculate_picp_for_intervals(y_test, percentile_predictions)
        # Convert the dictionary to a DataFrame
        picp_df = pd.DataFrame(list(picp_results.items()), columns=['Prediction Interval', 'PICP'])
        # Save the DataFrame to a CSV file
        picp_df.to_csv(os.path.join(sub_folder, 'picp_results.csv'), index=False)
        print('PICP results saved')

        # Calculate metrics
        r2 = r2_score(y_test, y_pred_50)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_50))
        mae = mean_absolute_error(y_test, y_pred_50)
        mean_error = np.mean(y_test - y_pred_50)

        # Calculate PICP for 90% prediction interval
        picp = np.mean((y_test >= y_pred_05) & (y_test <= y_pred_95))

        # Print metrics
        print(f'R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, ME: {mean_error:.2f}, PICP: {picp:.4f}')

        # Save the results (fraction, R², RMSE, MAE, ME, and PICP)
        results.append({
            'Fraction': frac, 
            'Training samples': len(X_train_sub),
            'R²': r2, 
            'RMSE': rmse, 
            'MAE': mae, 
            'ME': mean_error,
            'PICP': picp
        })

        print(f'Results saved for fraction {frac}.')

            
        # Copy the first 6 columns of test_data
        test_data_subset = soildepth_test.iloc[:, :6].copy()
        # Check if the index is intact
        print(test_data_subset.index.equals(test_data.index))  
        # Add the prediction results to the subset
        test_data_subset[f'y_pred_05_{frac}'] = y_pred_05
        test_data_subset[f'y_pred_10_{frac}'] = y_pred_10
        test_data_subset[f'y_pred_15_{frac}'] = y_pred_15
        test_data_subset[f'y_pred_20_{frac}'] = y_pred_20
        test_data_subset[f'y_pred_25_{frac}'] = y_pred_25
        test_data_subset[f'y_pred_30_{frac}'] = y_pred_30
        test_data_subset[f'y_pred_35_{frac}'] = y_pred_35
        test_data_subset[f'y_pred_40_{frac}'] = y_pred_40
        test_data_subset[f'y_pred_45_{frac}'] = y_pred_45
        test_data_subset[f'y_pred_50_{frac}'] = y_pred_50
        test_data_subset[f'y_pred_55_{frac}'] = y_pred_55
        test_data_subset[f'y_pred_60_{frac}'] = y_pred_60
        test_data_subset[f'y_pred_65_{frac}'] = y_pred_65
        test_data_subset[f'y_pred_70_{frac}'] = y_pred_70
        test_data_subset[f'y_pred_75_{frac}'] = y_pred_75
        test_data_subset[f'y_pred_80_{frac}'] = y_pred_80
        test_data_subset[f'y_pred_85_{frac}'] = y_pred_85
        test_data_subset[f'y_pred_90_{frac}'] = y_pred_90
        test_data_subset[f'y_pred_95_{frac}'] = y_pred_95

        # Save the predictions DataFrame to CSV
        test_data_subset.to_csv(os.path.join(sub_folder, f'test_predictions_{frac}.csv'), index=False)

        # # Run SHAP analysis only for fractions 1
        # if frac in [1.0]:
        #     # Plot and save the predicted vs observed plot
        #     plot_pred_vs_obs(y_pred_50, y_test, rmse, mean_error, sub_folder)
        #     print(f"Running SHAP analysis for fraction {frac}")
        #     shap_values = get_shap_values(qrf, X_test, quantile=0.5)
        #     shap_values_array = shap_values.values
        #     # Convert to a DataFrame
        #     shap_df = pd.DataFrame(shap_values_array, columns=X_test.columns)
        #     # Save to a CSV file
        #     shap_df.to_csv(os.path.join(sub_folder, 'shap_values.csv'), index=False)
        #     print("SHAP values saved as csv.")
        #     shap.summary_plot(shap_values.values, X_test, feature_names=X_test.columns)
        #     plt.savefig(os.path.join(sub_folder, f"SHAP_{frac}.png"), dpi=600)
        #     plt.close()
        #     print(f"SHAP plot saved for fraction {frac}.")


# Convert results to a dataframe and save it
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(sub_folder, 'subsampling_results.csv'), index=False)
    print("Predictions and test results saved to CSV.")
