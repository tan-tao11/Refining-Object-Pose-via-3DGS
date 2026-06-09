#!/usr/bin/env python
import cv2
import sys
import json
from pathlib import Path
import os

def downsize_mask(mask_parent_path):
    data_root = mask_parent_path
    mask_folder_path = os.path.join(data_root, 'masks')
    mask_files = os.listdir(mask_folder_path)
    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder_path, mask_file)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape[:2]
        processed_data_dir = data_root
        downscale_factors = [2, 4, 8]
        for downscale in downscale_factors:
            mask_path_i = os.path.join(processed_data_dir, f"masks_{downscale}")
            # processed_data_dir / f"masks_{downscale}"
            os.makedirs(mask_path_i, exist_ok=True)
            mask_path_i = os.path.join(mask_path_i, mask_file)
            # mask_path_i / "mask.png"
            mask_i = cv2.resize(
                mask, (width // downscale, height // downscale), interpolation=cv2.INTER_NEAREST
            )
            cv2.imwrite(str(mask_path_i), mask_i)
            print(f"Wrote {mask_path_i}")
    

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path_to/mask.png")
        # print(f"Output is path_to/masks_{downscale}/mask.png")
        sys.exit(1)
    data_root = str(sys.argv[1])
    mask_folder_path = os.path.join(data_root, 'masks')
    mask_files = os.listdir(mask_folder_path)
    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder_path, mask_file)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        height, width = mask.shape[:2]
        processed_data_dir = data_root
        downscale_factors = [2, 4, 8]
        for downscale in downscale_factors:
            mask_path_i = os.path.join(processed_data_dir, f"masks_{downscale}")
            # processed_data_dir / f"masks_{downscale}"
            os.makedirs(mask_path_i, exist_ok=True)
            mask_path_i = os.path.join(mask_path_i, mask_file)
            # mask_path_i / "mask.png"
            mask_i = cv2.resize(
                mask, (width // downscale, height // downscale), interpolation=cv2.INTER_NEAREST
            )
            cv2.imwrite(str(mask_path_i), mask_i)
            print(f"Wrote {mask_path_i}")