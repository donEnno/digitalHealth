#!/usr/bin/env python3

import os
import argparse

import nibabel as nib
import matplotlib.pyplot as plt

# Constants/defaults
DEFAULT_NIFTI_FILE = 'data/BraTS20_Validation_002_flair.nii'
DEFAULT_SLICE_INDEX = 99
DEFAULT_OUTPUT_FILE = 'flair.png'


def extract_and_store_slice(nifti_file: str, slice_index: int, output_file: str = DEFAULT_OUTPUT_FILE) -> None:
    """
    Extracts a specific slice from a NIfTI file and saves it as a PNG image.

    Args:
        nifti_file  (str): Path to the NIfTI file.
        slice_index (int): Index of the slice to extract.
        output_file (str): Path to save the output PNG file.
    """

    nii_object = nib.load(nifti_file)
    nii_data = nii_object.get_fdata()

    # Check if the slice index is valid
    if slice_index < 0 or slice_index >= nii_data.shape[2]:
        raise ValueError(f'Slice index {slice_index} is out of bounds for the data with shape {nii_data.shape}')
    
    # Extract the specified slice
    slice_data = nii_data[:, :, slice_index]

    # Save the slice as a PNG
    plt.imshow(slice_data, cmap='gray')
    plt.axis('off')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)        # tight layout to remove extra whitespace
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Load and and store a slice of a NIfTI file.')
    parser.add_argument('-f', '--file', type=str, default=DEFAULT_NIFTI_FILE, help='Path to the NIfTI file.')
    parser.add_argument('-s', '--slice', type=int, default=DEFAULT_SLICE_INDEX, help='Slice index to process.')
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT_FILE, help='Output PNG file name.')

    args = parser.parse_args()
    nifti_file = args.file
    slice_index = args.slice
    output_file = args.output

    # Check if the file exists
    if not os.path.isfile(nifti_file):
        raise FileNotFoundError(f'The file "{nifti_file}" does not exist.')
    
    extract_and_store_slice(nifti_file, slice_index, output_file)

    print(f'Slice {slice_index + 1} (index {slice_index}) extracted and saved as "{output_file}".')


if __name__ == '__main__':
    main()
