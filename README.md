

## Overview

This repository contains the code implementation for the research presented in "Depth to Bedrock Prediction and Uncertainty Assessment – A Quantile Regression Forest Approach to Map Unconsolidated Sediments in Sweden."

Research Abstract
Knowledge of depth to bedrock is key to better understanding Earth's structure. Its variability has implications for a range of fields, including soil science, geology, hydrology, and civil engineering. This study presents a high-resolution (10m) Quantile Regression Forest model for mapping Depth to bedrock across Sweden, comparing it against an existing national Inverse Distance Weighting model and a previously published global machine learning model. While the Inverse Distance Weighting approach showed lower overall error metrics, our Quantile Regression Forest model excelled at mapping shallow depths (0-10m), though it significantly underpredicted deeper ranges (>20m).
We utilized the Quantile Regression Forest algorithm's uncertainty quantification capabilities to assess prediction interval coverage across depth ranges and visualized uncertainty through both pixel-level maps and a novel QuadMap approach that encodes uncertainty via variable resolution. Our findings highlight the need for nuanced model evaluation beyond point estimates, particularly in heterogeneous glaciated landscapes, and suggest a hybrid approach for creating a more accurate Depth-to-bedrock map for Sweden.

Framework Capabilities
This framework provides a complete pipeline for depth to bedrock prediction using Quantile Random Forest regression, featuring:

Data loading and preprocessing of geological and topographic features
Training with various subsampling fractions to analyze data requirements
Hyperparameter optimization with Optuna for robust model tuning
Model evaluation with multiple metrics across different depth ranges
Prediction interval calibration (PICP) for uncertainty quantification
SHAP-based feature importance analysis for model interpretability
Comprehensive visualization and reporting of both predictions and uncertainty



## Project Structure

```
DTBMapping/
│
├── src/                           # Source modules
│   ├── __init__.py                # Makes src a proper package
│   ├── data_loader.py             # Data loading and preprocessing
│   ├── utils.py                   # Utility functions
│   ├── optimizer.py               # Hyperparameter optimization
│   ├── model_trainer.py           # Model training functions
│   ├── evaluator.py               # Metrics calculation and evaluation
│   └── plotter.py                 # Visualization functions
│
├── main.py                        # Main script that runs the pipeline
├── Dockerfile                     # dockerfile used to run the code, containing all required packages
└── README.md                      # This file
```

## Requirements

The following Python packages are required:

```
optuna
pandas
numpy
matplotlib
scikit-learn
quantile-forest
scipy
shap
joblib
tqdm
```

## Usage

### Basic Usage

To run the pipeline with default settings:

```bash
python main.py
```

This will:
1. Load training and testing data
2. Train models using different subsampling fractions
3. Optimize hyperparameters for each subsample
4. Generate predictions and evaluate models
5. Create visualizations and save results

### Configuration

To modify the data paths and other settings, edit the constants at the top of `main.py`:

```python
# Define data paths
TRAIN_DATA_PATH = '/path/to/your/training_data.csv'
TEST_DATA_PATH = '/path/to/your/testing_data.csv'
OUTPUT_BASE_FOLDER = '/path/to/your/output_directory'
```

### Customizing Subsampling Fractions

To change which subsampling fractions are used, modify the `subsample_fractions` list in `main.py`:

```python
# Define subsample fractions for training
subsample_fractions = [0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
```

### Adjusting Hyperparameter Optimization

To adjust the number of trials for Optuna optimization, modify the `n_optuna_trials` variable:

```python
n_optuna_trials = 20  # Increase for better hyperparameter search, decrease for faster execution
```

## Module Descriptions

### data_loader.py

Contains functions for loading and preprocessing soil depth data:

- `load_and_subset_data(train_path, test_path)`: Loads training and testing data, and subsets to relevant columns

### utils.py

Provides utility functions:

- `create_output_directory(output_folder)`: Creates a timestamped output directory
- `subsample_training_data(X_train, y_train, frac)`: Creates stratified subsamples of the training data

### optimizer.py

Handles hyperparameter optimization:

- `run_optimization(X, y, n_trials, output_path, frac)`: Runs Optuna optimization and saves results

### model_trainer.py

Manages model training:

- `train_and_save_model(X_train, y_train, best_params, sub_folder, frac)`: Trains and saves the QRF model

### evaluator.py

Contains evaluation functions:

- `calculate_metrics(y_true, y_pred, y_pred_lower, y_pred_upper)`: Calculates performance metrics
- `calculate_picp_for_intervals(y_test, percentile_predictions)`: Calculates Prediction Interval Coverage Probability
- `get_shap_values(qrf, X, quantile)`: Calculates SHAP values for model interpretation

### plotter.py

Handles visualization:

- `plot_pred_vs_obs(y_pred, y_test, rmse, mean_error, sub_folder, frac, train_size, test_size)`: Creates scatterplots of predicted vs. observed values
- `plot_shap_summary(shap_explanation, sub_folder, frac)`: Creates SHAP summary plots

## Output Files

The framework generates the following outputs in the timestamped directory:

- `subsampling_results.csv`: Summary metrics for all subsampling fractions
- `all_subsample_test_predictions.csv`: All predictions from all model runs
- `optuna_study_results_{frac}.csv`: Hyperparameter optimization results for each fraction
- `qrf_model_{frac}.pkl`: Saved model for each fraction
- `predicted_vs_observe_{frac}.png`: Scatter plots of predictions vs. observations
- `SHAP_{frac}.png`: SHAP summary plots for feature importance
- `shap_values_{frac}.csv`: SHAP values for detailed analysis

## Example Workflow

1. Data is loaded and preprocessed
2. For each subsampling fraction:
   - Training data is subsampled
   - Hyperparameters are optimized
   - Model is trained with best parameters
   - Model makes predictions across all quantiles
   - Metrics are calculated and stored
   - Visualizations are generated
3. All results are combined and saved
4. SHAP analysis is performed for selected fractions

## Extending the Framework

To add new features:

1. Create a new module in the `src` directory
2. Import it in `main.py`
3. Integrate your new functionality into the workflow

## Performance Considerations

- For very large datasets, consider increasing the subsampling step sizes
- Reduce `n_optuna_trials` for faster execution
- SHAP analysis is computationally intensive; limit to key fractions

## License

[Include your license information here]
