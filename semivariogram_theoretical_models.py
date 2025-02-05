import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.optimize import curve_fit
import seaborn as sns

def compute_semivariogram(coordinates, values, n_bins=20, max_distance=None):
    print("Starting semivariogram computation...")
    
    # Calculate the pairwise distances between coordinates
    pairwise_distances = pdist(coordinates)
    pairwise_values = pdist(values[:, np.newaxis])

    # Compute semivariance for each pair (half the squared difference of values)
    semivariances = 0.5 * pairwise_values ** 2

    # If max_distance is not set, use the maximum pairwise distance
    if max_distance is None:
        max_distance = np.max(pairwise_distances)

    # Create bins for distance intervals (this defines your lag distances)
    bins = np.linspace(0, max_distance, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Center of each bin for lag distances

    # Calculate the average semivariance for each distance bin
    semivariance_per_bin = []
    valid_bin_centers = []

    for i in range(n_bins):
        mask = (pairwise_distances >= bins[i]) & (pairwise_distances < bins[i + 1])
        if np.any(mask):
            semivariance_per_bin.append(np.mean(semivariances[mask]))
            valid_bin_centers.append(bin_centers[i])
        else:
            print(f"No data for bin centered at {bin_centers[i]:.2f}, skipping...")

    return np.array(valid_bin_centers), np.array(semivariance_per_bin)

# Define semivariogram models
def gaussian_model(h, nugget, sill, range_):
    return nugget + (sill - nugget) * (1 - np.exp(-(h ** 2) / (range_ ** 2)))

def exponential_model(h, nugget, sill, range_):
    return nugget + (sill - nugget) * (1 - np.exp(-h / range_))

def spherical_model(h, nugget, sill, range_):
    return np.where(
        h <= range_,
        nugget + ((sill - nugget) * (1.5 * (h / range_) - 0.5 * (h / range_)**3)),
        sill
    )

# Fit the specified semivariogram model to the empirical data
def fit_semivariogram_model(model_func, distances, semivariances):
    print(f"Fitting {model_func.__name__} model...")
    initial_params = [np.min(semivariances), np.max(semivariances), np.max(distances) / 3]
    params, _ = curve_fit(model_func, distances, semivariances, p0=initial_params)
    nugget, sill, range_ = params
    print(f"{model_func.__name__} model fitted. Nugget: {nugget:.2f}, Sill: {sill:.2f}, Range: {range_:.2f}")
    return nugget, sill, range_

# Load and sample data
print("Loading soil data...")
soiltraining = pd.read_csv('/workspace/data/soildepth/DepthData/csvs/new_train_jbas.csv', sep=',', decimal='.')

unique_values = soiltraining['jbas_merged'].unique()

print("Subsampling and computing semivariograms...")
annotations = []

for value in unique_values:
    print(f"Processing for '{value}'...")

    filtered_data = soiltraining[soiltraining['jbas_merged'] == value]

    if value == 'sjo':
        continue

    if len(filtered_data) > 100000 and len(filtered_data) <= 200000:
        subsample = filtered_data.sample(frac=0.5, random_state=42)
    elif len(filtered_data) > 200000 and len(filtered_data) <= 400000:
        subsample = filtered_data.sample(frac=0.3, random_state=42)
    elif len(filtered_data) > 400000:
        subsample = filtered_data.sample(frac=0.15, random_state=42)
    else:
        subsample = filtered_data

    coordinates = subsample[['N', 'E']].dropna().values
    values = subsample['DJUP'].dropna().values

    distances, semivariances = compute_semivariogram(coordinates, values, n_bins=50, max_distance=10000)

    models = {'Gaussian': gaussian_model, 'Exponential': exponential_model, 'Spherical': spherical_model}
    fitted_params = {}

    for model_name, model_func in models.items():
        try:
            nugget, sill, range_ = fit_semivariogram_model(model_func, distances, semivariances)
            fitted_params[model_name] = (nugget, sill, range_)

            # Create a separate plot for the current model
            plt.figure(figsize=(5, 5))
            sns.despine()
            plt.plot(distances, semivariances, 'x', markersize = 5, color = 'black', alpha = 0.8, label='Empirical Semivariogram')
            fitted_values = model_func(distances, *fitted_params[model_name])
            plt.plot(distances, fitted_values, color = 'darkgrey', label=f'{model_name} Model')
            plt.xlabel('Distance')
            plt.ylabel('Semivariance')
            plt.legend(loc='best')
            plt.grid()
            plt.title(f"{value}")

            # Annotate model parameters and data count on the plot
            annotation_text = (f"Nugget: {nugget:.2f}\nSill: {sill:.2f}\nRange: {range_:.2f}\nn: {len(subsample)}")
            plt.text(0.15, 0.2, annotation_text, transform=plt.gca().transAxes, 
                     fontsize=10, verticalalignment='top', bbox=dict(facecolor="white", alpha=0.7, edgecolor='none'))

            plot_filename = f'/workspace/data/soildepth/manuscript2plots/semivariogram/{value}_{model_name}.jpg'
            plt.savefig(plot_filename, dpi=600)
            plt.close()
            print(f"{model_name} semivariogram plot for '{value}' saved as {plot_filename}.")
        except RuntimeError as e:
            print(f"Model fitting failed for {model_name}: {e}")
            fitted_params[model_name] = None

    annotation_text = "\n".join(
        [
            f"{model_name}: Nugget={params[0]:.2f}, Sill={params[1]:.2f}, Range={params[2]:.2f}"
            for model_name, params in fitted_params.items() if params
        ]
    )

    num_points = len(subsample)

    annotation_text = f"Value: {value}\n" + annotation_text + f"\nn = {num_points}"
    annotations.append(annotation_text)

with open("/workspace/data/soildepth/manuscript2plots/semivariogram/annotations_with_values.txt", "w") as file:
    file.write("\n\n".join(annotations))

print("All semivariograms processed.")
