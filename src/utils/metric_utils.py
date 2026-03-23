import numpy as np
import os
import cv2
import torch
import os.path as osp
from loguru import logger
from time import time
from scipy import spatial
from src.utils.sample_points_on_cad import load_points_from_cad, model_diameter_from_bbox
from .colmap.read_write_model import qvec2rotmat
from .colmap.eval_helper import quaternion_from_matrix


def convert_pose2T(pose):
    # pose: [R: 3*3, t: 3]
    R, t = pose
    return np.concatenate(
        [np.concatenate([R, t[:, None]], axis=1), [[0, 0, 0, 1]]], axis=0
    )  # 4*4

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))

def projection_2d_error(model_3D_pts, pose_pred, pose_targets, K):
    def project(xyz, K, RT):
        """
        NOTE: need to use original K
        xyz: [N, 3]
        K: [3, 3]
        RT: [3, 4]
        """
        xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
        xyz = np.dot(xyz, K.T)
        xy = xyz[:, :2] / xyz[:, 2:]
        return xy

    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_targets.shape[0] == 4:
        pose_targets = pose_targets[:3]

    model_2d_pred = project(model_3D_pts, K, pose_pred) # pose_pred: 3*4
    model_2d_targets = project(model_3D_pts, K, pose_targets)
    proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))
    return proj_mean_diff

def add_metric(model_3D_pts, diameter, pose_pred, pose_target, percentage=0.1, syn=False, model_unit='m'):
    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_target.shape[0] == 4:
        pose_target = pose_target[:3]
    
    # if model_unit == 'm':
    #     model_3D_pts *= 1000
    #     diameter *= 1000
    #     pose_pred[:,3] *= 1000
    #     pose_target[:,3] *= 1000
        
    #     max_model_coord = np.max(model_3D_pts, axis=0)
    #     min_model_coord = np.min(model_3D_pts, axis=0)
    #     diameter_from_model = np.linalg.norm(max_model_coord - min_model_coord)
    # elif model_unit == 'mm':
    #     pass

    diameter_thres = diameter * percentage
    model_pred = np.dot(model_3D_pts, pose_pred[:, :3].T) + pose_pred[:, 3]
    model_target = np.dot(model_3D_pts, pose_target[:, :3].T) + pose_target[:, 3]
    
    if syn:
        mean_dist_index = spatial.cKDTree(model_pred)
        mean_dist, _ = mean_dist_index.query(model_target, k=1)
        mean_dist = np.mean(mean_dist)
    else:
        mean_dist = np.mean(np.linalg.norm(model_pred - model_target, axis=-1))
    if mean_dist < diameter_thres:
        return True
    else:
        return False


# Evaluate query pose errors
def query_pose_error(pose_pred, pose_gt, unit='m'):
    """
    Input:
    -----------
    pose_pred: np.array 3*4 or 4*4
    pose_gt: np.array 3*4 or 4*4
    """
    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_gt.shape[0] == 4:
        pose_gt = pose_gt[:3]

    # Convert results' unit to cm
    if unit == 'm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) * 100
    elif unit == 'cm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3])
    elif unit == 'mm':
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_gt[:, 3]) / 10
    else:
        raise NotImplementedError

    rotation_diff = np.dot(pose_pred[:, :3], pose_gt[:, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
    return angular_distance, translation_distance

import time
def ransac_PnP(
    K,
    pts_2d,
    pts_3d,
    scale=1,
    pnp_reprojection_error=5,
    img_hw=None,
    use_pycolmap_ransac=False,
):
    """ solve pnp """
    try:
        import pycolmap
    except:
        # logger.warning(f"pycolmap is not installed, use opencv ransacPnP instead")
        use_pycolmap_ransac = False

    if use_pycolmap_ransac:
        import pycolmap

        assert img_hw is not None and len(img_hw) == 2

        pts_2d = list(np.ascontiguousarray(pts_2d.astype(np.float64))[..., None]) # List(2*1)
        pts_3d = list(np.ascontiguousarray(pts_3d.astype(np.float64))[..., None]) # List(3*1)
        K = K.astype(np.float64)
        # Colmap pnp with non-linear refinement
        focal_length = K[0, 0]
        cx = K[0, 2]
        cy = K[1, 2]
        cfg = {
            "model": "SIMPLE_PINHOLE",
            "width": int(img_hw[1]),
            "height": int(img_hw[0]),
            "params": [focal_length, cx, cy],
        }

        ret = pycolmap.absolute_pose_estimation(
            pts_2d, pts_3d, cfg, max_error_px=float(pnp_reprojection_error)
        )
        qvec = ret["qvec"]
        tvec = ret["tvec"]
        pose_homo = convert_pose2T([qvec2rotmat(qvec), tvec])
        # Make inliers:
        inliers = ret['inliers']
        if len(inliers) == 0:
            inliers = np.array([]).astype(np.bool_)
        else:
            index = np.arange(0, len(pts_3d))
            inliers = index[inliers]

        return pose_homo[:3], pose_homo, inliers, True
    else:
        dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")

        pts_2d = np.ascontiguousarray(pts_2d.astype(np.float64))
        pts_3d = np.ascontiguousarray(pts_3d.astype(np.float64))
        K = K.astype(np.float64)

        pts_3d *= scale
        state = None
        try:
            time1 =time.time()
            _, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d,
                pts_2d,
                K,
                dist_coeffs,
                reprojectionError=pnp_reprojection_error,
                iterationsCount=10000,
                flags=cv2.SOLVEPNP_EPNP,
            )
            time2 =time.time()
            # print(time2-time1)
            rotation = cv2.Rodrigues(rvec)[0]

            tvec /= scale
            pose = np.concatenate([rotation, tvec], axis=-1)
            pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)

            if inliers is None:
                inliers = np.array([]).astype(np.bool_)
            state = True

            return pose, pose_homo, inliers, state
        except cv2.error:
            state = False
            return np.eye(4)[:3], np.eye(4), np.array([]).astype(np.bool_), state
        
def camera_2_world(viewmat):
    R_inv = viewmat[:3, :3]  # 3 x 3
    T_inv = viewmat[:3, 3:4]  # 3 x 1
    # Inverse of the inverse rotation matrix is the original rotation matrix
    R = R_inv
    # To get original translation, use the inverse of the analytic translation transformation
    # T = -R @ T_inv
    T = T_inv
    # flip the z and y axes back to the original alignment
    R_edit = np.diag(np.array([1, 1, -1], dtype=R.dtype))
    R = R @ R_edit
    R_edit_1 = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])
    R = R @ R_edit_1
    
    c2w = np.eye(4, dtype=R.dtype)
    c2w[:3, :3] = R
    c2w[:3, 3:4] = T

    return c2w

def world_2_camera(c2w):
    R = c2w[:3, :3]  # 3 x 3
    T = c2w[:3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R_edit = np.diag(np.array([1, -1, -1], dtype=R.dtype))
    R = R @ R_edit
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.T
    T_inv = -R_inv @ T
    viewmat = np.eye(4, dtype=R.dtype)
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv

    return viewmat

def draw_match(save_path, image1, image2, match_points_img1, match_points_img2):
    # 将两个图像水平拼接
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    new_height = max(height1, height2)
    new_width = width1 + width2
    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # 将灰度图像转换为BGR图像
    # image1_bgr = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    # image2_bgr = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    # 放置图像
    new_image[:height1, :width1] = image1
    new_image[:height2, width1:width1+width2] = image2
    
    # 生成多种颜色
    # num_matches = len(match_points_img1)
    # hsv_colors = [(i / num_matches, 1, 1) for i in range(num_matches)]
    # bgr_colors = [tuple(int(c * 255) for c in cv2.cvtColor(np.uint8([[[(h, s, v)]]]), cv2.COLOR_HSV2BGR)[0, 0]) for (h, s, v) in hsv_colors]
    
    bgr_colors = []
    for i in range(len(match_points_img1)):
        color = (int(255/len(match_points_img1)*(i+1)), int(255/len(match_points_img1)*(i+1)), int(255/len(match_points_img1)*(i+1)))
        bgr_colors.append(color)
        
    # 可视化匹配点和连线
    for (x1, y1), (x2, y2), color in zip(match_points_img1, match_points_img2, bgr_colors):
        if x2 >= 512 or y2 >= 512 or x2 <= 0 or y2 <= 0:
            continue
        cv2.circle(new_image, (int(x1), int(y1)), 5, color, -1)  # 绿色圆圈
        cv2.circle(new_image, (int(x2 + width1), int(y2)), 5, color, -1)  # 绿色圆圈
        cv2.line(new_image, (int(x1), int(y1)), (int(x2 + width1), int(y2)), color, 2)  # 蓝色连线
        
    cv2.imwrite(save_path, new_image)
    
def project(X, T, K):
    # convert the common coordinates into homogeneous coordinates
    X_hom = np.hstack((X, np.ones((X.shape[0], 1))))
    T = T[:3, :]
    P = K@T
    x = (P@X_hom.T).T
    x_normalized = x[:, :2] / x[:, 2][:, np.newaxis]
    depths = x[:, 2]
    
    return x_normalized, depths

def gt_convert(gt_pose):
    R_inv = gt_pose[:3, :3]  # 3 x 3
    T_inv = gt_pose[:3, 3:4]  # 3 x 1
    # Inverse of the inverse rotation matrix is the original rotation matrix
    R = R_inv
    # To get original translation, use the inverse of the analytic translation transformation
    # T = -R @ T_inv
    T = T_inv
    # flip the z and y axes back to the original alignment
    R_edit_1 = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])
    R = R @ R_edit_1
    R_edit = np.diag(np.array([1, 1, -1], dtype=R.dtype))
    R = R @ R_edit
    
    c2w = np.eye(4, dtype=R.dtype)
    c2w[:3, :3] = R
    c2w[:3, 3:4] = T

    return c2w
    
def draw_matchs(img_path, gt_pose, K, mkpts_3d, mkpts_query_f, inliers=None):
    # gt_pose_cam = gt_convert(gt_pose)
    gt_pose_cam = gt_pose
    x, _ = project(mkpts_3d, gt_pose_cam, K)
    draw_match('match_all.png', cv2.imread(img_path[0]), cv2.imread(img_path[0]), mkpts_query_f, x)
    if inliers is not None and inliers.shape[0] > 0:
        x = x[inliers][:, 0, :]
        mkpts_query_f = mkpts_query_f[inliers][:, 0, :]
    
    draw_match('match.png', cv2.imread(img_path[0]), cv2.imread(img_path[0]), mkpts_query_f, x)
    

import time
@torch.no_grad()
def compute_query_pose_errors(
    data, configs, training=False
):
    """
    Update:
        data(dict):{
            "R_errs": []
            "t_errs": []
            "inliers": []
        }
    """
    model_unit = configs['model_unit'] if 'model_unit' in configs else 'm'

    time1 = time.time()
    m_bids = data["m_bids"].cpu().numpy()
    mkpts_3d = data["mkpts_3d_db"].cpu().numpy()
    mkpts_query = data["mkpts_query_f"].cpu().numpy()
    img_orig_size = (
        torch.tensor(data["q_hw_i"]).numpy() * data["query_image_scale"].cpu().numpy()
    )  # B*2
    query_K = data["query_intrinsic"].cpu().numpy()
    query_pose_gt = data["query_pose_gt"].cpu().numpy()  # B*4*4

    data.update({"R_errs": [], "t_errs": [], "inliers": []})
    data.update({"R_errs_c": [], "t_errs_c": [], "inliers_c": []})

    # Prepare query model for eval ADD metric
    if 'eval_ADD_metric' in configs:
        if configs['eval_ADD_metric'] and not training:
            image_path = data['query_image_path']
            adds = True if ('0810-' in image_path) or ('0811-' in image_path) else False # Symmetric object in LINEMOD
            query_K_origin = data["query_intrinsic_origin"].cpu().numpy()
            model_path = osp.join(image_path.rsplit('/', 3)[0], 'model_eval.ply')
            if not osp.exists(model_path):
                model_path = osp.join(image_path.rsplit('/', 3)[0], 'model.ply')
            diameter_file_path = osp.join(image_path.rsplit('/', 3)[0], 'diameter.txt')
            if not osp.exists(model_path):
                logger.error(f'want to eval add metric, however model_eval.ply path:{model_path} not exists!')
            else:
                # Load model:
                model_vertices, bbox = load_points_from_cad(model_path) # N*3
                # Load diameter:
                if osp.exists(diameter_file_path):
                    diameter = np.loadtxt(diameter_file_path)
                else:
                    diameter = model_diameter_from_bbox(bbox)
                
                data.update({"ADD":[], "proj2D":[]})

    time2 = time.time()
    pose_pred = []
    for bs in range(query_K.shape[0]):
        mask = m_bids == bs

        mkpts_query_f = mkpts_query[mask]
        query_pose_pred, query_pose_pred_homo, inliers, state = ransac_PnP(
            query_K[bs],
            mkpts_query_f,
            mkpts_3d[mask],
            scale=configs["point_cloud_rescale"],
            img_hw=img_orig_size[bs].tolist(),
            pnp_reprojection_error=configs["pnp_reprojection_error"],
            use_pycolmap_ransac=configs["use_pycolmap_ransac"],
        )
        time3 = time.time()
        draw_matchs(data["query_image_path"], query_pose_gt[bs], query_K[bs], mkpts_3d, mkpts_query_f, inliers)
        # query_pose_pred_homo = camera_2_world(query_pose_pred_homo)
        query_pose_pred = query_pose_pred_homo[:3, :]
        pose_pred.append(query_pose_pred_homo)
        
        if query_pose_pred is None:
            data["R_errs"].append(np.inf)
            data["t_errs"].append(np.inf)
            data["inliers"].append(np.array([]).astype(np.bool_))
            if "ADD" in data:
                data['ADD'].append(False)
        else:
            R_err, t_err = query_pose_error(query_pose_pred, query_pose_gt[bs], unit=model_unit)
            data["R_errs"].append(R_err)
            data["t_errs"].append(t_err)
            data["inliers"].append(inliers)

            if "ADD" in data:
                add_result = add_metric(model_vertices, diameter, pose_pred=query_pose_pred, pose_target=query_pose_gt[bs], syn=adds)
                data["ADD"].append(add_result)

                proj2d_result = projection_2d_error(model_vertices, pose_pred=query_pose_pred, pose_targets=query_pose_gt[bs], K=query_K_origin[bs])
                data['proj2D'].append(proj2d_result)

    time4 = time.time()
    pose_pred = np.stack(pose_pred)  # [B*4*4]
    data.update({"pose_pred": pose_pred})
    


def aggregate_metrics(metrics, pose_thres=[1, 3, 5], proj2d_thres=5):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    """
    R_errs = metrics["R_errs"]
    t_errs = metrics["t_errs"]

    agg_metric = {}
    for pose_threshold in pose_thres:
        agg_metric[f"{pose_threshold}cm@{pose_threshold}degree"] = np.mean(
            (np.array(R_errs) < pose_threshold) & (np.array(t_errs) < pose_threshold)
        )

    if "ADD_metric" in metrics:
        ADD_metric = metrics['ADD_metric']
        agg_metric["ADD metric"] = np.mean(ADD_metric)

        proj2D_metric = metrics['proj2D_metric']
        agg_metric["proj2D metric"] = np.mean(np.array(proj2D_metric) < proj2d_thres)

    return agg_metric
