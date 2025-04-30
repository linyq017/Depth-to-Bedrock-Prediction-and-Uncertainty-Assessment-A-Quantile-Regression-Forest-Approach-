import pandas as pd
import os
import numpy as np
import joblib # Import joblib to save predictions dataframe

# Import functions from our new modules
from src.data_loader import load_and_subset_data
from src.utils import create_output_directory, subsample_training_data
from src.optimizer import run_optimization
from src.model_trainer import train_and_save_model
from src.evaluator import calculate_metrics, calculate_picp_for_intervals, get_shap_values
from src.plotter import plot_pred_vs_obs, plot_shap_summary

# Define data paths (replace with your actual paths)
TRAIN_DATA_PATH = '/workspace/data/soildepth/DepthData/csvs/training_faultline.csv'
TEST_DATA_PATH = '/workspace/data/soildepth/DepthData/csvs/testing_faultline.csv'
OUTPUT_BASE_FOLDER = '/workspace/data/soildepth/Hypertune'

if __name__ == '__main__':
    # Create the main output directory and a timestamped subfolder
    sub_folder = create_output_directory(OUTPUT_BASE_FOLDER)

    # Load and prepare data
    train_data, test_data_subsetted, original_test_data = load_and_subset_data(TRAIN_DATA_PATH, TEST_DATA_PATH)

    # Split features and target
    X_train = train_data.drop(columns='DJUP')
    y_train = train_data['DJUP']
    X_test = test_data_subsetted.drop(columns='DJUP')
    y_test = test_data_subsetted['DJUP']

    print(f'Original X_train size: {len(X_train)}, y_train size: {len(y_train)}')
    print(f'X_test size: {len(X_test)}, y_test size: {len(y_test)}')
    print('X_train columns :', X_train.columns.tolist())

    # Define subsample fractions for training
    # Ensure 1.0 is included if you want to train on the full dataset
    subsample_fractions = [0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]#
    # List to store results for different fractions
    results_list = []
    all_predictions = [] # To store predictions for all fractions in one DataFrame

    # Copy the first few columns of original_test_data once
    # This assumes the first 6 columns are consistent across runs/subsets
    # Make sure these columns don't change based on subsampling.
    # If 'DJUP', 'N', 'E', etc. are in the first 6, and 'DJUP' is the target,
    # maybe copy the original test data excluding DJUP, or just specific identifier columns.
    # Let's copy the original test data's index and coordinates if available.
    # Assuming 'N' and 'E' are good identifiers.
    # If your test file structure is consistent, copying the first 6 columns *of the original file* might be what you intended.
    # Let's rely on the index matching between test_data_subsetted and original_test_data.
    # The index should be preserved if subset_columns just selects columns.
    base_test_info = original_test_data.iloc[:, :6].copy()


    for frac in subsample_fractions:
        print(f"\n--- Processing subsample fraction: {frac} ---")

        # Subsample the training data
        X_train_sub, y_train_sub = subsample_training_data(X_train, y_train, frac=frac)
        print(f'Subsampled X_train size: {len(X_train_sub)}, y_train size: {len(y_train_sub)}')

        # Hyperparameter optimization using the subsampled data
        # We run fewer trials for smaller subsamples as hyperparameter tuning precision might be lower
        n_optuna_trials = 2 # Adjust based on available resources and time
        study = run_optimization(X_train_sub, y_train_sub, n_trials=n_optuna_trials, output_path=sub_folder, frac=frac)
        best_params = study.best_params
        print(f"Best hyperparameters for fraction {frac}: {best_params}")

        # Train the final model on the subsampled data with best parameters
        qrf_model = train_and_save_model(X_train_sub, y_train_sub, best_params, sub_folder, frac=frac)

        # Make predictions on the *full* test set using the trained model
        print(f"Making predictions on the full test set for fraction {frac}...")
        quantiles_to_predict = [
            0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
            0.5, # Median prediction
            0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
        ]
        y_pred_quantiles = qrf_model.predict(X_test, quantiles=quantiles_to_predict)

        # Extract specific quantiles
        quantile_map = {q: i for i, q in enumerate(quantiles_to_predict)}
        y_pred_50 = y_pred_quantiles[:, quantile_map[0.50]]
        y_pred_05 = y_pred_quantiles[:, quantile_map[0.05]]
        y_pred_95 = y_pred_quantiles[:, quantile_map[0.95]]

        # Store all predictions for this fraction in a DataFrame
        preds_df_frac = base_test_info.copy() # Start with base test info
        # Ensure the index aligns before adding prediction columns
        preds_df_frac = preds_df_frac.loc[X_test.index] # Align with the index of X_test
        for i, q in enumerate(quantiles_to_predict):
             preds_df_frac[f'y_pred_{int(q*100):02d}_{frac}'] = y_pred_quantiles[:, i]

        # Append to the list of all predictions
        all_predictions.append(preds_df_frac)


        # Calculate core metrics using the 50th percentile prediction
        metrics = calculate_metrics(y_test, y_pred_50, y_pred_05, y_pred_95)

        # Calculate PICP for multiple intervals
        percentile_intervals = {
             '10': (y_pred_quantiles[:, quantile_map[0.45]], y_pred_quantiles[:, quantile_map[0.55]]), # 10% PI (45th-55th)
             '20': (y_pred_quantiles[:, quantile_map[0.40]], y_pred_quantiles[:, quantile_map[0.60]]), # 20% PI (40th-60th)
             '30': (y_pred_quantiles[:, quantile_map[0.35]], y_pred_quantiles[:, quantile_map[0.65]]), # 30% PI (35th-65th)
             '40': (y_pred_quantiles[:, quantile_map[0.30]], y_pred_quantiles[:, quantile_map[0.70]]), # 40% PI (30th-70th)
             '50': (y_pred_quantiles[:, quantile_map[0.25]], y_pred_quantiles[:, quantile_map[0.75]]), # 50% PI (25th-75th)
             '60': (y_pred_quantiles[:, quantile_map[0.20]], y_pred_quantiles[:, quantile_map[0.80]]), # 60% PI (20th-80th)
             '70': (y_pred_quantiles[:, quantile_map[0.15]], y_pred_quantiles[:, quantile_map[0.85]]), # 70% PI (15th-85th)
             '80': (y_pred_quantiles[:, quantile_map[0.10]], y_pred_quantiles[:, quantile_map[0.90]]), # 80% PI (10th-90th)
             '90': (y_pred_quantiles[:, quantile_map[0.05]], y_pred_quantiles[:, quantile_map[0.95]])  # 90% PI (5th-95th)
        }
        picp_results = calculate_picp_for_intervals(y_test, percentile_intervals)

        # Add PICP results to the metrics dictionary
        metrics.update({f'PICP_{k}%': v for k, v in picp_results.items()})

        # Store results for this fraction
        fraction_results = {
            'Fraction': frac,
            'Training samples': len(X_train_sub),
            'Test samples': len(y_test),
            **metrics # Include all calculated metrics
        }
        results_list.append(fraction_results)
        print(f"Metrics calculated and stored for fraction {frac}.")

        # Generate plots only for specific fractions if needed (e.g., full data)
        if frac in [0.001, 0.01,0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ]: # 
             plot_pred_vs_obs(y_pred_50, y_test, metrics['RMSE'], metrics['ME'], sub_folder, frac, len(X_train_sub), len(y_test))

             # Run SHAP analysis only for specific fractions if needed
             if frac in [1.0]: # Choose fractions for SHAP
                 try:
                     # Get SHAP values for the 50th percentile
                     shap_explanation = get_shap_values(qrf_model, X_test, quantile=0.5)
                     # Save SHAP values to CSV
                     shap_df = pd.DataFrame(shap_explanation.values, columns=X_test.columns)
                     shap_df['base_value'] = shap_explanation.base_values # Include the adjusted base value
                     shap_df.to_csv(os.path.join(sub_folder, f'shap_values_{frac}.csv'), index=False)
                     print(f"SHAP values saved for fraction {frac}.")

                     # Plot SHAP summary
                     plot_shap_summary(shap_explanation, sub_folder, frac)

                 except Exception as e:
                     print(f"Error during SHAP calculation or plotting for fraction {frac}: {e}")


    # Combine all prediction DataFrames
    if all_predictions:
         # Combine horizontally, aligning by index
         combined_predictions_df = pd.concat(all_predictions, axis=1)

         # Ensure original test data columns are also included in the final prediction file
         # Keep only the columns from original_test_data that are not feature columns in X_test
         # and are not already added as predictions.
         # A safer approach might be to merge based on index.
         # Let's merge the base_test_info with the combined predictions.
         # Since base_test_info was created from original_test_data and aligned by index
         # with X_test's index, a direct merge should work.
         # Ensure the index name is set if needed for merging.
         # combined_predictions_df.index.name = 'original_index' # Example
         # base_test_info.index.name = 'original_index' # Example

         # Merge base_test_info (which includes original index and first few columns)
         # with the combined predictions based on their index.
         final_predictions_df = base_test_info.merge(combined_predictions_df, left_index=True, right_index=True, how='left')


         # Save the combined predictions DataFrame to CSV
         predictions_output_path = os.path.join(sub_folder, 'all_subsample_test_predictions.csv')
         final_predictions_df.to_csv(predictions_output_path, index=False) # Set index=False if you don't want the index in the CSV
         print(f"Combined test predictions for all fractions saved to {predictions_output_path}.")


    # Convert results list to a dataframe and save it
    results_df = pd.DataFrame(results_list)
    results_output_path = os.path.join(sub_folder, 'subsampling_results.csv')
    results_df.to_csv(results_output_path, index=False)
    print(f"Subsampling results summary saved to {results_output_path}.")

    print("\n--- Script finished ---")