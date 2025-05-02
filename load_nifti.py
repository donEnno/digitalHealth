import os
import argparse

import nibabel as nib
import matplotlib.pyplot as plt

#!/usr/bin/env python3


def extract_and_store_slice(nifti_file, slice_index):

    nii_object = nib.load(nifti_file)
    nii_data = nii_object.get_fdata()

    # Check if the slice index is valid
    if slice_index < 0 or slice_index >= nii_data.shape[2]:
        raise ValueError(f'Slice index {slice_index} is out of bounds for the data with shape {nii_data.shape}')
    
    # Extract the specified slice
    slice_data = nii_data[:, :, slice_index]

    # Save the slice as a png
    plt.imshow(slice_data, cmap='gray')
    plt.axis('off')
    plt.savefig(f'./flair.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Load and process NIfTI files.')
    parser.add_argument('-f', '--file', type=str, default='data/BraTS20_Validation_002_flair.nii', help='Path to the NIfTI file.')
    parser.add_argument('-s', '--slice', type=int, default=99, help='Slice index to process.')
    
    args = parser.parse_args()
    nifti_file = args.file
    slice_index = args.slice

    # Check if the file exists
    if not os.path.isfile(nifti_file):
        raise FileNotFoundError(f'The file {nifti_file} does not exist.')
    
    extract_and_store_slice(nifti_file, slice_index)

    print(f'{slice_index+1}th slice at index {slice_index} extracted and saved as flair.png')


if __name__ == '__main__':
    main()
