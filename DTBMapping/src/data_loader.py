import pandas as pd
import os

def load_and_subset_data(train_path, test_path):
    """
    Loads training and testing data and subsets relevant columns.

    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the testing CSV file.

    Returns:
        tuple: A tuple containing:
            - train_data (pd.DataFrame): Subsetted training data.
            - test_data (pd.DataFrame): Subsetted testing data.
    """
    print("Loading and subsetting data...")
    soildepth_train = pd.read_csv(train_path, sep=',', decimal='.')
    soildepth_test = pd.read_csv(test_path, sep=',', decimal='.')

    relevant_columns = [
        'DJUP', 'N', 'E', 'Aspect50', 'ProCur50', 'RTP20_20', 'RTP50_50', 'Slope20', 'DEM',
        'EAS1ha', 'DI2m', 'CVA', 'SDFS', 'DFME', 'Rugged', 'HKDepth', 'MSRM', 'MED',
        'jbas_merged_grus', 'jbas_merged_hall', 'jbas_merged_isalvssediment',
        'jbas_merged_lera', 'jbas_merged_moran', 'jbas_merged_sand', 'jbas_merged_sjo',
        'jbas_merged_torv', 'karttyp_2', 'karttyp_3', 'karttyp_4', 'karttyp_5', 'karttyp_6',
        'karttyp_7', 'karttyp_8', 'karttyp_9', 'tekt_n_0', 'tekt_n_67', 'tekt_n_68',
        'tekt_n_69', 'tekt_n_70', 'tekt_n_72', 'tekt_n_79', 'tekt_n_82', 'tekt_n_88',
        'tekt_n_337', 'tekt_n_346', 'tekt_n_368', 'tekt_n_380', 'tekt_n_387', 'tekt_n_388',
        'tekt_n_389', 'tekt_n_390', 'tekt_n_394', 'tekt_n_1939', 'Geomorphon_Flat',
        'Geomorphon_Footslope', 'Geomorphon_Hollow(concave)', 'Geomorphon_Peak(summit)',
        'Geomorphon_Pit(depression)', 'Geomorphon_Ridge', 'Geomorphon_Shoulder',
        'Geomorphon_Slope', 'Geomorphon_Spur(convex)', 'Geomorphon_Valley',
        'DistanceToDeformation'
    ]

    train_data = soildepth_train[relevant_columns].copy()
    test_data = soildepth_test[relevant_columns].copy()

    print("Data loaded and subsetted successfully.")
    return train_data, test_data, soildepth_test # Return original test_data as well for later use

if __name__ == '__main__':
    # Example usage:
    # Replace with your actual paths
    train_file = '/workspace/data/soildepth/DepthData/csvs/training_faultline.csv'
    test_file = '/workspace/data/soildepth/DepthData/csvs/testing_faultline.csv'
    train_df, test_df, _ = load_and_subset_data(train_file, test_file)
    print("Train data shape:", train_df.shape)
    print("Test data shape:", test_df.shape)