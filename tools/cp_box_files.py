import os
import shutil

def copy_box3d_to_3dgs(root_dir="dataset_local/OnePose/test_data"):
    """Copy ``box3d_corners.txt`` from each object folder into its ``3DGS`` subfolder.

    Args:
        root_dir: Root directory containing per-object subdirectories.
    """
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)

        if not os.path.isdir(subdir_path):
            continue

        src_file = os.path.join(subdir_path, "box3d_corners.txt")
        if not os.path.exists(src_file):
            print(f"Warning: box3d_corners.txt not found under {subdir_path}")
            continue

        dst_dir = os.path.join(subdir_path, "3DGS")
        os.makedirs(dst_dir, exist_ok=True)

        dst_file = os.path.join(dst_dir, "box3d_corners.txt")
        shutil.copy2(src_file, dst_file)
        print(f"Copied: {src_file} -> {dst_file}")


if __name__ == "__main__":
    copy_box3d_to_3dgs()
    print("Done.")