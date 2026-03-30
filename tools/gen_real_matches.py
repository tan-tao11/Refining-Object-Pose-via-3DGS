import cv2
import torch
import numpy as np
import argparse
import os
import glob
import open3d as o3d
import shutil
import json
import torch.nn.functional as F
from scipy.ndimage import minimum_filter
from .utils.network import build_backbone
from tqdm import tqdm
from threadpoolctl import threadpool_limits




def read_gs_ckpt(ckpt_path, threshold=0.5, max_num_gaussians = 20000):
    # Load checkpoint file
    data = torch.load(ckpt_path, map_location='cpu')
    splats = data['splats']
    
    opacities = splats['opacities']
    opacities = torch.sigmoid(opacities)
    
    # Cull gaussians with low opacity
    culls = (opacities > threshold).squeeze()
    
    # If there are too many valid gaussians, only select the first n gaussians
    if culls.sum() > max_num_gaussians:
        opa_valid = opacities[culls]
        opa_valid_sorted, _ = torch.sort(opa_valid, descending=True)
        threshold = opa_valid_sorted[max_num_gaussians]
        culls = (opacities > threshold).squeeze()
    
    # Get means and descriptors
    descriptors = np.array(splats['gs_features'][culls])
    means = np.array(splats['means'][culls])
    
    return descriptors, means

def project_points_to_image(points, pose, camera_intrinsics, image_width, image_height):
    """
    将三维点投影到二维图像平面上，并找到每个像素点对应的最近的三维点序号

    参数:
    points (np.ndarray): 三维点云 (N, 3)
    camera_intrinsics (np.ndarray): 相机内参 (3, 3)
    image_width (int): 图像宽度
    image_height (int): 图像高度

    返回:
    np.ndarray: 每个像素点对应的最近的三维点序号 (image_height, image_width)
    """
    # 投影点云到图像平面
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    projected_points = camera_intrinsics @ pose[:3, :] @ points_homogeneous.T
    projected_depth = projected_points[2]
    projected_points = projected_points[:2] / projected_points[2]

    # 3. 过滤出投影在图像内部的点
    x, y = projected_points.astype(np.int32)  # (N,)
    valid_mask = (0 <= x) & (x < image_width) & (0 <= y) & (y < image_height) & (projected_depth > 0)

    x_valid, y_valid, depth_valid = x[valid_mask], y[valid_mask], projected_depth[valid_mask]
    indices_valid = np.nonzero(valid_mask)[0]  # 保留的 3D 点索引

    # 4. 使用 NumPy 计算每个像素的最近点
    distances = np.full((image_height, image_width), np.inf, dtype=np.float32)
    closest_indices = np.full((image_height, image_width), -1, dtype=np.int32)

    update_mask = depth_valid < distances[y_valid, x_valid]  # 仅更新更近的点
    distances[y_valid[update_mask], x_valid[update_mask]] = depth_valid[update_mask]
    closest_indices[y_valid[update_mask], x_valid[update_mask]] = indices_valid[update_mask]

    # # 创建距离矩阵并初始化为无穷大
    # distances = np.full((image_height, image_width), np.inf)
    # closest_indices = np.full((image_height, image_width), -1)

    # # 遍历所有投影点，更新距离矩阵和最近点索引矩阵
    # for i, (x, y) in enumerate(projected_points.T):
    #     x, y = int(x), int(y)
    #     if 0 <= x < image_width and 0 <= y < image_height:
    #         z = projected_depth[i]  # 使用点的深度作为距离度量
    #         if z < distances[y, x]:
    #             distances[y, x] = z
    #             closest_indices[y, x] = i

    return closest_indices, distances

def convert_indices(closest_indices, distances):
    """
    将 closest_indices 从二维转换为一维，并去除没有找到对应点的元素，同时给出对应的二维点序号

    参数:
    closest_indices (np.ndarray): 每个像素点对应的最近的三维点序号 (image_height, image_width)

    返回:
    np.ndarray: 有效的三维点序号 (一维)
    np.ndarray: 有效点的二维坐标 (N, 2)
    np.ndarray: 有效点的线性索引 (N)
    """
    height, width = closest_indices.shape
    indices_2d = np.array(np.nonzero(closest_indices >= 0)).T
    valid_indices = closest_indices[indices_2d[:, 0], indices_2d[:, 1]]
    valid_distances = distances[indices_2d[:, 0], indices_2d[:, 1]]
    linear_indices = np.ravel_multi_index((indices_2d[:, 0], indices_2d[:, 1]), (height, width))
    
    return valid_indices, indices_2d[:, [1, 0]], linear_indices, valid_distances

def create_depth_image(coords, depths, image_width, image_height):
    """
    根据二维坐标和深度值创建深度图
    参数:
        coords (np.ndarray): 二维坐标列表 (N, 2)
        depths (np.ndarray): 深度值列表 (N,)
        image_width (int): 图像宽度
        image_height (int): 图像高度
    返回:
        np.ndarray: 深度图 (image_height, image_width)
    """
    # 初始化深度图，填充为无穷大表示初始状态
    depth_image = np.full((image_height, image_width), np.inf)

    # 遍历所有坐标和深度值，将深度值填入对应位置
    for (x, y), depth in zip(coords, depths):
        if 0 <= x < image_width and 0 <= y < image_height:
            depth_image[y, x] = depth

    return depth_image

def filter_depth(depth, closest_indices):
    window_size = 5
    threshold = 0.01
    # 定义一个大小为3x3的过滤器，找到局部极小值
    local_min = minimum_filter(depth, size=window_size)

    # 创建一个与深度图像相同大小的数组，初始值为无穷大
    filtered_image = np.full(depth.shape, np.inf)
            
    # 只有当该点的深度值为局部极小值时才保留深度值
    filtered_image[depth == local_min] = depth[depth == local_min]
    
    height, width = closest_indices.shape
    indices_2d = np.array(np.nonzero(filtered_image <= 10)).T
    valid_indices = closest_indices[indices_2d[:, 0], indices_2d[:, 1]]
    return filtered_image, valid_indices, indices_2d[:, [1, 0]]

def get_gt_des_loftr(image, loftr_backbone):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = torch.from_numpy(image).cuda().to(torch.float32)[None, None, :, :]
    H, W = image.shape[2], image.shape[3]

    with torch.no_grad():
        output = loftr_backbone(image)

        feat_c = F.interpolate(output[0], size=(H, W), mode='nearest').permute(0, 2, 3, 1)
        feat_f = F.interpolate(output[1], size=(H, W), mode='nearest').permute(0, 2, 3, 1)

        descriptor_image = torch.cat([feat_c, feat_f], dim=3)
        
    return descriptor_image[0]

def des_filter(descriptors_img, descriptors_3d, indices_2d, valid_indices, threshold = 0.7):
    # 获取对应像素点的特征
    y_coords = indices_2d[:, 1]
    x_coords = indices_2d[:, 0]
    descriptors_px = descriptors_img[y_coords, x_coords]
    
    # 获取三维点的特征
    descriptors_3d = torch.from_numpy(descriptors_3d[valid_indices]).cuda()
    
    # 计算余弦相似度
    similarities = F.cosine_similarity(descriptors_3d, descriptors_px, dim=1)
    # similarities = 1 - F.l1_loss(descriptors_px, descriptors_3d, dim=1)
    
    # 筛选相似度高于阈值的序号
    selected_indices = torch.nonzero(similarities > threshold, as_tuple=False).squeeze()
    
    # 选择相似度高于阈值的匹配关系
    valid_indices = valid_indices[selected_indices.cpu()]
    indices_2d = indices_2d[selected_indices.cpu()]
    
    return valid_indices, indices_2d

def points_filter(points, valid_indices_all):
    # 一维的bool数组记录三维点投影请求，false为没有投影到任何一张图像
    pro_flag = np.zeros(points.shape[0], dtype=bool)
    
    # 遍历投影序号valid_indices_all，更新pro_flag
    for valid_indices in valid_indices_all:
        pro_flag[valid_indices] = True
    
    # 找出pro_flag中为True的序号，对应的三维点投影到了图像中
    old_idxs = np.where(pro_flag)[0]
    
    # 构建旧序号到新序号的映射字典
    old_to_new_idx = {old_idxs[new_idx]: new_idx for new_idx in range(old_idxs.shape[0])}
    
    new_valid_indices_all = []
    for valid_indices in valid_indices_all:
        new_valid_indices = [old_to_new_idx[old_idx] for old_idx in valid_indices]
        new_valid_indices_all.append(new_valid_indices)
    
    return old_idxs, new_valid_indices_all

def filter_by_bbox(bbox, points, indices_2d, valid_indices):
    points_valid = points[valid_indices]
    filter_box = bbox
    min_bound = filter_box.min(axis=0)  # The minimize bounding
    max_bound = filter_box.max(axis=0)  # The maxmize bouding
    
    if_filter = np.any((points_valid < min_bound) | (points_valid > max_bound), axis=1)
    
    if_valid = ~if_filter

    valid_indices = valid_indices[if_valid]
    indices_2d = indices_2d[if_valid]
    
    return valid_indices, indices_2d
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate real matches")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--ckpt_root", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    parser.add_argument("--data_type", type=str, default='train', help='train | val')
    parser.add_argument("--opacity_thresold", type=float, default=0.9)
    parser.add_argument("--max_gaussians", type=int, default=50000)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--descriptor_filter", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--depth_filter", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--descriptor_filter_thr", type=float, default=0.7)
    
    with threadpool_limits(limits=10, user_api="blas"):
        # num_threads = 1
        # os.environ.update({
        #     "OMP_NUM_THREADS": str(num_threads),
        #     "OPENBLAS_NUM_THREADS": str(num_threads),
        #     "MKL_NUM_THREADS": str(num_threads),
        #     "VECLIB_MAXIMUM_THREADS": str(num_threads),
        #     "NUMEXPR_NUM_THREADS": str(num_threads),
        #     # "TORCH_NCCL_BLOCKING_WAIT": "0"
        # })
        # torch.set_num_threads(num_threads)

        args = parser.parse_args()
        data_dir = args.data
        ckpt_root = args.ckpt_root
        opacity_thresold = args.opacity_thresold
        max_gaussians = args.max_gaussians
        data_type = args.data_type
        interval = args.interval
        descriptor_filter = args.descriptor_filter
        depth_filter = args.depth_filter
        save_root = args.save
        descriptor_filter_thr = args.descriptor_filter_thr
        
        # If use descriptor filter, build the feature extractor
        if descriptor_filter:
            net = build_backbone('ResNetFPN_8_2', 'weight/LoFTR_wsize9.ckpt')
            
        # Preprocess each object sequence in order
        sequences = os.listdir(data_dir)
        sequences.sort(key=lambda x: int(x.split('-')[0]))
        
        for sequence in tqdm(sequences, desc="sequences", ncols=80):
            try:
                sequence_path = os.path.join(data_dir, sequence)
                ckpt_folder = os.path.join(ckpt_root, sequence, 'ckpts')
                ckpt_paths = glob.glob(f'{ckpt_folder}/*.pt')
                ckpt_paths.sort(key=lambda x:int(x.split('/')[-1].split("_")[1]))
                ckpt_path = ckpt_paths[-1]
                print(f'Read 3DGS Splats from {ckpt_path}.')
                
                # Read checkpoint file
                descriptors_3d, means = read_gs_ckpt(ckpt_path, opacity_thresold, max_gaussians)

                # preprocess each folder
                obj_folders = os.listdir(sequence_path)
                obj_folders = [obj_folder for obj_folder in obj_folders if '-' in obj_folder]
                obj_folders.sort(key=lambda x: int(x.split("-")[-1]))
                
                # Output folder
                output_anno = os.path.join(save_root, sequence, 'anno')
                if os.path.exists(output_anno):
                    shutil.rmtree(output_anno)
                os.makedirs(output_anno)
                
                counter = 0
                anno_loftr_gs_file_all = []
                indices_2d_all = []
                valid_indices_all = []
                anno2d = []
                num_matchs = []
                num_images = 0
                
                # For training data, use all folders. For testing data, use the first folders
                if data_type == 'train':
                    obj_folders = obj_folders
                elif data_type == 'val':
                    obj_folders = obj_folders[:1]
                else:
                    raise TypeError
                for obj_folder in obj_folders:
                    num_images += len(os.listdir(os.path.join(sequence_path, obj_folder, 'color')))
                pbar_sequence = tqdm(total=num_images/interval, desc="sequence", ncols=100)
                
                for obj_folder in obj_folders:
                    # Color, Pose, Camera folder
                    img_folder_path = os.path.join(sequence_path, obj_folder, 'color')
                    pose_folder_path = os.path.join(sequence_path, obj_folder, 'poses_ba')
                    intr_folder_path = os.path.join(sequence_path, obj_folder, 'intrin_ba')
                    
                    # Color, Pose, Camera files
                    pose_files =os.listdir(pose_folder_path)
                    pose_files.sort(key = lambda x : int(x.split(".")[0]))
                    intrinsic_files = os.listdir(intr_folder_path)
                    intrinsic_files.sort(key = lambda x : int(x.split(".")[0]))
                    img_files = os.listdir(img_folder_path)
                    img_files.sort(key = lambda x : int(x.split(".")[0]))
                    
                    # Path to save result
                    anno_loftr_gs_dir = os.path.join(sequence_path, obj_folder, 'anno_loftr_gs')
                    if os.path.exists(anno_loftr_gs_dir):
                        shutil.rmtree(anno_loftr_gs_dir)
                    os.makedirs(anno_loftr_gs_dir)
                    
                    for pose_file, intrinsic_file, img_file in zip(pose_files[::interval], intrinsic_files[::interval], img_files[::interval]):
                        try:
                            # Skip invalid files
                            if not pose_file.split('.')[-1] == 'txt' or not intrinsic_file.split('.')[-1] == 'txt':
                                continue 
                            
                            pose_file_path = os.path.join(pose_folder_path, pose_file)
                            intrinsic_file_path = os.path.join(intr_folder_path, intrinsic_file)
                            img_file_path = os.path.join(img_folder_path, img_file)
                            
                            pose = np.loadtxt(pose_file_path)
                            intr = np.loadtxt(intrinsic_file_path)
                            img = cv2.imread(img_file_path)
                            
                            width, height = img.shape[1], img.shape[0]
                            
                            # Project points onto image and map each pixel to a point indices 
                            indices, distances = project_points_to_image(means, pose, intr, width, height)
                            valid_indices, indices_2d, linear_indices, depth = convert_indices(indices, distances)
                            
                            if False:
                                def draw_points(img, points, save_img, radius=1, color=(0, 255, 0), thickness=1):
                                    for point in points:
                                        x, y = point[0], point[1]
                                        cv2.circle(img, (int(x), int(y)), radius, color, thickness)
                                    cv2.imwrite(save_img, img)
                                draw_points(img, indices_2d, 'vis_valid_pixels.png')
                            
                            # Filter out invalid correspondings
                            if depth_filter:
                                depth_img = create_depth_image(indices_2d, depth, width, height)
                                depth_filtered, valid_indices, indices_2d = filter_depth(depth_img, indices)
                                
                            if True:
                                bbox = np.loadtxt(os.path.join(sequence_path, "box3d_corners.txt",))
                                valid_indices, indices_2d = filter_by_bbox(bbox, means, indices_2d, valid_indices)
                                
                            if descriptor_filter:
                                descriptors_image = get_gt_des_loftr(img_file_path, net)
                                valid_indices, indices_2d = des_filter(descriptors_image, descriptors_3d, indices_2d, valid_indices, descriptor_filter_thr)
                            
                            if len(valid_indices)<=10:
                                continue
                            
                            # Save result
                            anno_loftr_gs_file = os.path.join(anno_loftr_gs_dir, pose_file.replace('txt', 'json'))
                            anno_loftr_gs_file_all.append(anno_loftr_gs_file)
                            indices_2d_all.append(indices_2d)
                            valid_indices_all.append(valid_indices)
                            
                            # Save annotation information
                            anno2d_img = {"anno_id": counter, "anno_file": anno_loftr_gs_file, "img_file": img_file_path, "pose_file": pose_file_path}
                            anno2d.append(anno2d_img)
                            # print("{}: {}".format(i, counter))
                            num_matchs.append(len(valid_indices))
                            
                            # Vis valid points
                            if False:
                                # 创建点云对象
                                matched_points = means[valid_indices]
                                point_cloud = o3d.geometry.PointCloud()
                                point_cloud.points = o3d.utility.Vector3dVector(matched_points)

                                # 保存为PLY文件
                                o3d.io.write_point_cloud(f"valid_points_{sequence}.ply", point_cloud)
                            
                            pbar_sequence.update(1)
                            counter += 1
                        except Exception as e:
                            print(f'  Warning: {img_file} failed: {e}')
                            continue
                        
                # Filter out 3D points
                old_idxs, new_valid_indices_all = points_filter(means, valid_indices_all)
                
                # Save 2D annotations
                for i in range(len(anno_loftr_gs_file_all)):
                    anno_loftr_gs_file = anno_loftr_gs_file_all[i]
                    valid_indices = new_valid_indices_all[i]
                    indices_2d = indices_2d_all[i]
                    assign_matrix = [[i for i in range(len(valid_indices))]]
                    assign_matrix.append(list(map(int, valid_indices)))
                    data_2d = {'keypoints2d': indices_2d.astype(int).tolist(), 'scores2d': np.ones(indices_2d.shape[0], dtype=float).tolist(), 'assign_matrix': assign_matrix}
                    with open(anno_loftr_gs_file, 'w') as f:
                        json.dump(data_2d, f)    

                # Save 3D points and descriptors
                points = means[old_idxs]
                des_c_all = descriptors_3d[:, :256]
                des_f_all = descriptors_3d[:, 256:]
                des_f = des_f_all[old_idxs]
                des_c = des_c_all[old_idxs]
                output_3d_file = os.path.join(output_anno, 'anno_3d_average.npz')
                scores = np.ones([len(points), 1])
                np.savez(output_3d_file, keypoints3d=points, descriptors3d=des_f.T, scores3d=scores)
                # 粗特征
                output_3d_coarse_file = os.path.join(output_anno, 'anno_3d_average_coarse.npz')
                np.savez(output_3d_coarse_file, keypoints3d=points, descriptors3d=des_c.T, scores3d=scores)
            
                # Save all valid points in a ply file
                if True:
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(points)
                    o3d.io.write_point_cloud("output/saved_points/{}_src.ply".format(sequence), point_cloud)   
                
                # Save 2D annotation information
                anno2d_obj_file = os.path.join(output_anno, 'anno_2d.json')
                with open(anno2d_obj_file, 'w') as f:
                    json.dump(anno2d, f)
                    print(f'{sequence} down! length :{len(anno2d)}')
            except:
                print(f'Error for {sequence}')