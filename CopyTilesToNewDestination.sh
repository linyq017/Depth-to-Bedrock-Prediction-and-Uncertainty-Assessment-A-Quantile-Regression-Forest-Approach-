#!/bin/bash

# Source directory (where your files or subfolders are located)
SOURCE_DIR="/workspace/data/soildepth/Indices/Indices_Tiles/DistanceToDeformationZones"

# Destination directory (where files will be copied)
DEST_DIR="/workspace/data/soildepth/Indices/TestTiles"

# File containing the list of filenames
FILE_LIST="/workspace/data/krycklan/tif_names.txt"

# Option to copy only from the top-level directory
FLAT_COPY=false

# Parse optional argument
if [[ "$1" == "--flat" ]]; then
    FLAT_COPY=true
    echo "Flat copy mode enabled: Only copying files from $SOURCE_DIR (not subfolders)."
fi

# Ensure the destination directory exists
mkdir -p "$DEST_DIR"

# Check if the file list exists
if [[ ! -f "$FILE_LIST" ]]; then
    echo "Error: File list $FILE_LIST not found!"
    exit 1
fi

echo "Starting file copy process..."

if $FLAT_COPY; then
    # Copy only from the top-level directory
    while IFS= read -r FILENAME; do
        FILE="$SOURCE_DIR/$FILENAME"

        echo "Checking: $FILE"  # Debugging output

        if [[ -f "$FILE" ]]; then
            cp "$FILE" "$DEST_DIR/"
            echo "Copied: $FILE -> $DEST_DIR/"
        else
            echo "File not found: $FILE"
        fi
    done < "$FILE_LIST"
else
    # Loop through all subdirectories in the source directory
    find "$SOURCE_DIR" -type d | while read -r SUBFOLDER; do
        # Extract the relative path of the subfolder
        REL_PATH="${SUBFOLDER#$SOURCE_DIR/}"

        echo "Processing folder: $SUBFOLDER (relative path: $REL_PATH)"

        # Create the corresponding subfolder in the destination directory
        mkdir -p "$DEST_DIR/$REL_PATH"

        # Read filenames from the file and copy matching ones
        while IFS= read -r FILENAME; do
            FILE="$SUBFOLDER/$FILENAME"

            echo "Checking: $FILE"  # Debugging output

            if [[ -f "$FILE" ]]; then
                cp "$FILE" "$DEST_DIR/$REL_PATH/"
                echo "Copied: $FILE -> $DEST_DIR/$REL_PATH/"
            else
                echo "File not found: $FILE"
            fi
        done < "$FILE_LIST"
    done
fi

echo "Files have been copied to $DEST_DIR"
