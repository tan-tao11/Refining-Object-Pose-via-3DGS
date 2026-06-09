import os 
import shutil
import argparse
import warnings
import cv2
import os.path as osp
import numpy as np
from tqdm import tqdm
from .mask.predict_masks import predict_masks
from .mask.predict_masks_fastsam import predict_masks_fastsam
from .mask.predict_masks_mobilesam import predict_masks_mobilesam
from .mask.downsize_mask import downsize_mask

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

def pose_transform(pose):
    R = pose[:3, :3]
    t = pose[:3, 3]

    matrix = R
    w = np.sqrt(1 + matrix[0, 0] + matrix[1, 1] + matrix[2, 2]) / 2
    # Extract the vector components (x, y, z)
    x = (matrix[2, 1] - matrix[1, 2]) / (4 * w)
    y = (matrix[0, 2] - matrix[2, 0]) / (4 * w)
    z = (matrix[1, 0] - matrix[0, 1]) / (4 * w)
    quaternion = np.array([w, x, y, z])

    return quaternion, t

def preprocess_val(data_path, save_path, text_prompt, sam_model='sam', interval=1, image_w=512, image_h=512):
    sub_folders = os.listdir(data_path)
    sequences = []
    for folder in sub_folders:
        if '-' in folder:
            sequences.append(folder)
    sequences.sort(key=lambda x: int(x.split("-")[1]))
    
    full_images_path = osp.join(save_path, 'images')
    os.makedirs(full_images_path)

    full_masks_path = osp.join(save_path, 'masks')
    os.makedirs(full_masks_path)

    sfm_save_path = osp.join(save_path, 'sparse/0')
    os.makedirs(sfm_save_path)

    with open(osp.join(sfm_save_path, 'points3D.txt'), 'w') as f:
        pass

    id_last = 0
    for sequence in sequences[:1]:
        seq_path = osp.join(data_path, sequence)
        image_folder_path = osp.join(seq_path, 'color')
        intr_folder_path = osp.join(seq_path, 'intrin_ba')
        pose_folder_path = osp.join(seq_path, 'poses_ba')

        # Predict masks for images
        mask_folder_path = osp.join(seq_path, 'mask')
        if osp.exists(mask_folder_path):
            shutil.rmtree(mask_folder_path)
        os.makedirs(mask_folder_path)
        if sam_model == 'sam':
            predict_masks(image_folder_path, text_prompt, mask_folder_path, interval=interval)
        elif sam_model == "fast_sam":
            predict_masks_fastsam(image_folder_path, text_prompt, mask_folder_path)
        elif sam_model == "mobile_sam":
            predict_masks_mobilesam(image_folder_path, text_prompt, mask_folder_path)
        else:
            raise NotImplementedError

        pose_files = os.listdir(pose_folder_path)
        pose_files = [f for f in pose_files if f.endswith('txt')]
        pose_files.sort(key=lambda x: int(x.split('.')[0]))

        images_txt = osp.join(sfm_save_path, 'images.txt')
        camera_txt  = os.path.join(sfm_save_path, 'cameras.txt')
        with open(images_txt, 'a+') as f:
            with open(camera_txt, 'a+') as f_c:
                id = id_last
                for pose_file in pose_files[::interval]:
                    pose_file_path = osp.join(pose_folder_path, pose_file)
                    pose = np.loadtxt(pose_file_path)
                    quat, t = pose_transform(pose)

                    content = str(id)+' '+' '.join(map(str, quat))+' '+' '.join(map(str, t))+' {}'.format(str(id)) \
                            +' {}'.format(str(id)+'.png')+'\n\n'
                    
                    image_src_path = osp.join(image_folder_path, pose_file.replace("txt", "png"))
                    image_des_path = osp.join(full_images_path, str(id)+'.png')
                    shutil.copy(image_src_path, image_des_path)

                    mask_src_path = osp.join(mask_folder_path, pose_file.replace("txt", "png"))
                    mask_des_path = osp.join(full_masks_path, str(id)+'.png')
                    try:
                        shutil.copy(mask_src_path, mask_des_path)
                    except:
                        mask = np.ones([image_h, image_w, 1], dtype=np.uint8) * 255
                        cv2.imwrite(mask_des_path, mask)

                    f.write(content)

                    intrinsic_file_path = osp.join(intr_folder_path, pose_file)
                    intr = np.loadtxt(intrinsic_file_path)
                    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
                    content = str(id)+' PINHOLE {} {} {} {} {} {}\n'.format(image_w, image_h, fx, fy, cx, cy)
                    f_c.write(content)

                    id += 1
                id_last = id
    # Convert TXT to BIN
    os.system('colmap model_converter --input_path {} --output_path {} --output_type {}'\
              .format(sfm_save_path, sfm_save_path, 'BIN'))     
    
def preprocess(data_path, save_path, text_prompt, sam_model='sam', interval=1, image_w=512, image_h=512):
    sub_folders = os.listdir(data_path)
    sequences = []
    for folder in sub_folders:
        if '-' in folder:
            sequences.append(folder)
    sequences.sort(key=lambda x: int(x.split("-")[1]))

    full_images_path = osp.join(save_path, 'images')
    os.makedirs(full_images_path)

    full_masks_path = osp.join(save_path, 'masks')
    os.makedirs(full_masks_path)

    sfm_save_path = osp.join(save_path, 'sparse/0')
    os.makedirs(sfm_save_path)

    with open(osp.join(sfm_save_path, 'points3D.txt'), 'w') as f:
        pass

    id_last = 0
    for sequence in sequences:
        seq_path = osp.join(data_path, sequence)
        image_folder_path = osp.join(seq_path, 'color')
        intr_folder_path = osp.join(seq_path, 'intrin_ba')
        pose_folder_path = osp.join(seq_path, 'poses_ba')

        # Predict masks for images
        mask_folder_path = osp.join(seq_path, 'mask')
        if osp.exists(mask_folder_path):
            shutil.rmtree(mask_folder_path)
        os.makedirs(mask_folder_path)
        if sam_model == 'sam':
            predict_masks(image_folder_path, text_prompt, mask_folder_path, interval=interval)
            # predict_masks(image_folder_path, text_prompt, mask_folder_path)
        elif sam_model == "fast_sam":
            predict_masks_fastsam(image_folder_path, text_prompt, mask_folder_path)
        elif sam_model == "mobile_sam":
            predict_masks_mobilesam(image_folder_path, text_prompt, mask_folder_path)
        else:
            raise NotImplementedError

        pose_files = os.listdir(pose_folder_path)
        pose_files = [f for f in pose_files if f.endswith('txt')]
        pose_files.sort(key=lambda x: int(x.split('.')[0]))

        images_txt = osp.join(sfm_save_path, 'images.txt')
        camera_txt  = os.path.join(sfm_save_path, 'cameras.txt')
        with open(images_txt, 'a+') as f:
            with open(camera_txt, 'a+') as f_c:
                id = id_last
                for pose_file in pose_files[::interval]:
                    pose_file_path = osp.join(pose_folder_path, pose_file)
                    pose = np.loadtxt(pose_file_path)
                    quat, t = pose_transform(pose)

                    content = str(id)+' '+' '.join(map(str, quat))+' '+' '.join(map(str, t))+' {}'.format(str(id)) \
                            +' {}'.format(str(id)+'.png')+'\n\n'
                    
                    image_src_path = osp.join(image_folder_path, pose_file.replace("txt", "png"))
                    image_des_path = osp.join(full_images_path, str(id)+'.png')
                    shutil.copy(image_src_path, image_des_path)

                    mask_src_path = osp.join(mask_folder_path, pose_file.replace("txt", "png"))
                    mask_des_path = osp.join(full_masks_path, str(id)+'.png')
                    try:
                        shutil.copy(mask_src_path, mask_des_path)
                    except:
                        mask = np.ones([image_h, image_w, 1], dtype=np.uint8) * 255
                        cv2.imwrite(mask_des_path, mask)

                    f.write(content)

                    intrinsic_file_path = osp.join(intr_folder_path, pose_file)
                    intr = np.loadtxt(intrinsic_file_path)
                    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
                    content = str(id)+' PINHOLE {} {} {} {} {} {}\n'.format(image_w, image_h, fx, fy, cx, cy)
                    f_c.write(content)

                    id += 1
                id_last = id
    # Convert TXT to BIN
    os.system('colmap model_converter --input_path {} --output_path {} --output_type {}'\
              .format(sfm_save_path, sfm_save_path, 'BIN'))        

    # output_dir = data_path
    # image_folder = os.path.join(output_dir, "images")
    # output_dir_mask = os.path.join(output_dir, 'masks')
    # predict_masks(image_folder,  text_prompt, output_dir_mask)
    # downsize_mask(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data.')
    parser.add_argument("--data_root", type=str, required=True, help="dataset root")
    parser.add_argument("--interval", type=int, default=5, help="interval for preprocess data")
    parser.add_argument("--sam_model", type=str, default="sam", help="sam | fast_sam")
    parser.add_argument("--data_type", type=str, default="train", help="train | val")

    args = parser.parse_args()

    data_root = args.data_root
    interval = args.interval
    sam_model = args.sam_model
    data_type = args.data_type

    data_sequences = os.listdir(data_root)
    data_sequences.sort()

    for seq in data_sequences:
        if not '-' in seq:
            continue
        print(seq)

        data_path = osp.join(data_root, seq)
        save_path = osp.join(data_path, '3DGS')
        if osp.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)

        prompt_list = seq.split('-')[1:]
        text_prompt = ''
        for prompt in prompt_list:
            text_prompt += prompt + ' '
        # text_prompt = 'cookies box'
        if data_type == "train":
            preprocess(data_path,  save_path, text_prompt, sam_model=sam_model, interval=interval)
        else:
            preprocess_val(data_path,  save_path, text_prompt, sam_model=sam_model, interval=interval)
        

        