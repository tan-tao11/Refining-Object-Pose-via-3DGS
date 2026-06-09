import cv2 
import os
import argparse
import shutil
import os.path as osp

def downsample_image(image_path, save_path, downsample_factor):
    """Downsample an image by an integer factor and write to ``save_path``."""
    if not isinstance(downsample_factor, int) or downsample_factor <= 0:
        raise ValueError("downsample_factor must be a positive integer.")

    image = cv2.imread(image_path, -1)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    height, width = image.shape[:2]
    new_width = width // downsample_factor
    new_height = height // downsample_factor

    downsampled_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    cv2.imwrite(save_path, downsampled_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downsample images.')
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--factor", type=int, default=2)

    args = parser.parse_args()

    data_dir = args.data_dir
    factor = args.factor

    data_sequences = os.listdir(data_dir)
    data_sequences.sort()

    for seq in data_sequences:
        if not '-' in seq:
            continue
        print(seq)

        image_folder_src = osp.join(data_dir, seq, '3DGS/images')
        image_folder_dst = osp.join(data_dir, seq, f'3DGS/images_{factor}')
        mask_folder_dst = image_folder_dst.replace("images", "masks")
        if osp.exists(image_folder_dst):
            shutil.rmtree(image_folder_dst)
        os.makedirs(image_folder_dst)

        if osp.exists(mask_folder_dst):
            shutil.rmtree(mask_folder_dst)
        os.makedirs(mask_folder_dst)

        image_files = os.listdir(image_folder_src)
        image_files = [f for f in image_files if f.endswith('png')]

        for image_file in image_files:
            image_file_src = osp.join(image_folder_src, image_file)
            image_file_dst = osp.join(image_folder_dst, image_file)

            mask_file_src = image_file_src.replace("images", "masks")
            mask_file_dst = image_file_dst.replace("images", "masks")

            downsample_image(image_file_src, image_file_dst, factor)
            downsample_image(mask_file_src, mask_file_dst, factor)
