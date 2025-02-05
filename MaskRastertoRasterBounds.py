import os
import glob
import rasterio as rio
from rasterio.mask import mask
from shapely.geometry import box
from rasterio.coords import BoundingBox
import argparse

def mask_raster_with_bounds(input_raster, reference_raster_folder, output_base_folder):
    """
    Masks a raster or all rasters in a folder using the bounds of a reference raster
    and saves the clipped rasters with the same names in an output folder.
    
    Args:
        input_raster (str or list): A single raster file or a list of raster files to be masked.
        reference_raster_folder (str): Folder containing the reference raster whose bounds will be used.
        output_base_folder (str): Base folder where the output folder will be created.
        
    Returns:
        None
    """
    # Ensure the input is a list of rasters (single raster or a folder of rasters)
    if isinstance(input_raster, str):
        # If a single raster, wrap it in a list
        if os.path.isdir(input_raster):
            # If input is a folder, get all rasters in that folder
            input_raster_files = glob.glob(os.path.join(input_raster, '*.tif'))
        else:
            # If a single raster file, use it directly
            input_raster_files = [input_raster]
    elif isinstance(input_raster, list):
        input_raster_files = input_raster
    else:
        raise ValueError("input_raster must be either a file path or a list of raster files.")

    # Get all raster files in the reference raster folder to extract the bounds
    reference_raster_files = glob.glob(os.path.join(reference_raster_folder, '*.tif'))
    
    if not reference_raster_files:
        raise ValueError("No raster files found in the reference raster folder.")
    
    # Use the first raster from the folder to get the bounds
    with rio.open(reference_raster_files[0]) as ref_src:
        bbox = ref_src.bounds  # Get the bounds of the reference raster
        
    # Convert the bounding box to a Shapely box (geometry) for masking
    bounds = BoundingBox(left=bbox.left, bottom=bbox.bottom, right=bbox.right, top=bbox.top)
    mask_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    
    # Process each input raster file
    for input_raster_path in input_raster_files:
        # Extract the input raster folder name for creating an output subfolder
        input_raster_folder = os.path.basename(os.path.dirname(input_raster_path))
        
        # Create the output folder based on the input raster's folder name
        output_folder = os.path.join(output_base_folder, input_raster_folder)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"Processing raster: {input_raster_path}")
        
        # Open the input raster
        with rio.open(input_raster_path) as src:
            # Mask the raster with the geometry (bounding box of the reference raster)
            out_image, out_transform = mask(src, [mask_geom], crop=True)
            out_meta = src.meta.copy()  # Copy the metadata
            
            # Update the metadata for the new raster
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # Define the output file path (use the same name as input raster)
            output_path = os.path.join(output_folder, os.path.basename(input_raster_path))
            
            # Write the masked raster to the output folder
            with rio.open(output_path, 'w', **out_meta) as dst:
                dst.write(out_image, 1)
            
            print(f"Masked raster saved to: {output_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Mask raster(s) with the bounds of a reference raster.")
    
    # Command-line arguments
    parser.add_argument('input_raster', type=str, help="Path to a single raster or a folder of rasters")
    parser.add_argument('reference_raster_folder', type=str, help="Folder containing the reference raster")
    parser.add_argument('output_base_folder', type=str, help="Base folder where output rasters will be saved")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function to mask rasters
    mask_raster_with_bounds(args.input_raster, args.reference_raster_folder, args.output_base_folder)

if __name__ == "__main__":
    main()
