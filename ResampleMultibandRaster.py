import os
import argparse
import rasterio
from rasterio import Affine

try:
    import whitebox
    wbt = whitebox.WhiteboxTools()
except:
    from WBT.whitebox_tools import WhiteboxTools
    wbt = WhiteboxTools()

BAND_NAMES = ['DEM', 'EAS1ha', 'EAS 10ha', 'DI2m', 'CVA', 'SDFS', 'DFME', 'Rugged', 'NMD ', 'SoilMap',
              'HKDepth', 'SoilDepth', 'LandAge', 'MSRM', 'Northing', 'Easting', 'MED']

def save_error_log(file_path, error_file):
    """Save the name of the file that failed processing to a text file."""
    with open(error_file, "a") as log_file:
        log_file.write(f"{file_path}\n")

def get_processed_files(output_folder_base):
    """
    Generate a set of already processed file names based on existing subfolders.
    Assumes that the output folder structure is based on BAND_NAMES.
    """
    processed_files = set()
    for band_name in BAND_NAMES:
        band_folder = os.path.join(output_folder_base, band_name)
        if os.path.exists(band_folder):
            for file in os.listdir(band_folder):
                if file.endswith(".tif"):
                    processed_files.add(file)  # Add only the file name
    return processed_files

def resample_and_save(img_path, output_folder_base, processed_files, cell_size=10, error_file="error_log.txt"):
    """
    Resample each band in a multiband raster and save the output
    to separate folders named after the predefined band names.
    """
    img_name = os.path.basename(img_path)
    if img_name in processed_files:
        print(f"Skipping already processed file: {img_name}")
        return

    try:
        with rasterio.open(img_path) as src:
            if src.count != len(BAND_NAMES):
                raise ValueError(f"Number of bands ({src.count}) does not match the predefined band names ({len(BAND_NAMES)}).")

            meta = src.meta
            meta.update({
                'driver': 'GTiff',
                'dtype': src.dtypes[0],
                'transform': Affine(cell_size, 0, src.bounds.left, 0, -cell_size, src.bounds.top),
                'width': int((src.bounds.right - src.bounds.left) / cell_size),
                'height': int((src.bounds.top - src.bounds.bottom) / cell_size)
            })

            for band_idx, band_name in enumerate(BAND_NAMES, start=1):
                band_data = src.read(band_idx)
                output_folder = os.path.join(output_folder_base, band_name)
                os.makedirs(output_folder, exist_ok=True)
                resample_path = os.path.join(output_folder, img_name)

                with rasterio.open(resample_path, 'w', **meta) as dst:
                    dst.write(band_data, 1)

                print(f"Resampled Band {band_name} saved to {resample_path}")

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        save_error_log(img_path, error_file)

def main(input_path, output_path_resample, cell_size=10, error_file="error_log.txt"):
    if not os.path.exists(input_path):
        raise ValueError('Input path does not exist: {}'.format(input_path))

    # Get already processed files
    processed_files = get_processed_files(output_path_resample)

    # Collect files to process
    if os.path.isdir(input_path):
        imgs = sorted(
            [os.path.join(input_path, f) for f in os.listdir(input_path)
            if f.endswith('.tif') and os.path.splitext(f)[0].isdigit()],
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
    else:
        imgs = [input_path]

    # Process each image
    for img_path in imgs:
        print(f"Processing: {img_path}")
        resample_and_save(img_path, output_path_resample, processed_files, cell_size, error_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract topographical indices for image(s)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_path', help='Path to DEM or folder of DEMs')
    parser.add_argument('output_path_resample', help='Directory to store resampled images')
    parser.add_argument('--cell_size', type=float, default=10, help='Cell size for resampling')
    parser.add_argument('--error_file', type=str, default="error_log.txt", help='Path to save the error log')
    args = vars(parser.parse_args())
    main(**args)
