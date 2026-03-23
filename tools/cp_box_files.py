import os
import shutil

def copy_box3d_to_3dgs(root_dir="dataset_local/OnePose/test_data"):
    """
    将每个子文件夹中的 box3d_corners.txt 复制到 3DGS 子文件夹中
    :param root_dir: 根目录，默认为 "train_data"
    """
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        
        # 确保是文件夹
        if not os.path.isdir(subdir_path):
            continue
        
        # 查找 box3d_corners.txt
        src_file = os.path.join(subdir_path, "box3d_corners.txt")
        if not os.path.exists(src_file):
            print(f"警告: {subdir_path} 中没有找到 box3d_corners.txt")
            continue
        
        # 创建 3DGS 文件夹（如果不存在）
        dst_dir = os.path.join(subdir_path, "3DGS")
        os.makedirs(dst_dir, exist_ok=True)
        
        # 复制文件
        dst_file = os.path.join(dst_dir, "box3d_corners.txt")
        shutil.copy2(src_file, dst_file)
        print(f"已复制: {src_file} -> {dst_file}")

if __name__ == "__main__":
    copy_box3d_to_3dgs()
    print("操作完成！")