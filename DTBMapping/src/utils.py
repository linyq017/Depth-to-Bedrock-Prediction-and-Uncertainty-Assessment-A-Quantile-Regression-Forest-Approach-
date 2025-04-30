import os
from datetime import datetime
import pandas as pd

def create_output_directory(output_folder):
    """
    Creates a timestamped subfolder within the specified output folder.

    Args:
        output_folder (str): The base directory for output.

    Returns:
        str: The path to the newly created subfolder.
    """
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    sub_folder = os.path.join(output_folder, timestamp)
    os.makedirs(sub_folder, exist_ok=True)
    print(f"Created output subfolder at {sub_folder}.")
    return sub_folder

def subsample_training_data(X_train, y_train, frac=1.0):
    """
    Subsamples the training data.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        frac (float, optional): The fraction of data to sample. Defaults to 1.0.

    Returns:
        tuple: A tuple containing:
            - X_train_sub (pd.DataFrame): Subsampled training features.
            - y_train_sub (pd.Series): Subsampled training target.
    """
    if frac < 1.0:
        print(f"Subsampling {frac * 100:.2f}% of the training data...")
        X_train_sub = X_train.sample(frac=frac, random_state=42) # Added random_state for reproducibility
        y_train_sub = y_train.loc[X_train_sub.index]
    else:
        print("Using 100% of the training data (no subsampling).")
        X_train_sub = X_train
        y_train_sub = y_train

    return X_train_sub, y_train_sub

if __name__ == '__main__':
    # Example usage:
    output_dir = create_output_directory('./output_tests')
    print(f"Output directory: {output_dir}")

    # Dummy data for subsampling test
    X_dummy = pd.DataFrame({'col1': range(100), 'col2': range(100)})
    y_dummy = pd.Series(range(100))

    X_sub, y_sub = subsample_training_data(X_dummy, y_dummy, frac=0.5)
    print("Original data size:", len(X_dummy))
    print("Subsampled data size:", len(X_sub))