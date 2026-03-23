import torch
import json
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation 

class OnePoseDataset(Dataset):
    def __init__(
        self, 
        anno_file,
        sigma_rot=0.1,
        sigma_trans=0.05,
        ):
        super(Dataset, self).__init__()

        self.sigma_rot = sigma_rot
        self.sigma_trans = sigma_trans
        # Load merged annotation file
        with open(anno_file, 'r') as f:
            self.annos = json.load(f)
            
    def add_noise_to_pose(self, R_original, t_original, sigma_rot, sigma_trans):
        """
        根据指定的旋转和平移误差，生成带有噪声的新姿态。

        参数:
        - R_original: 原始旋转矩阵 (3x3 numpy array)
        - t_original: 原始平移向量 (3x1 numpy array)
        - sigma_rot: 旋转噪声的标准差（弧度）
        - sigma_trans: 平移噪声的标准差

        返回:
        - R_new: 带有噪声的旋转矩阵 (3x3 numpy array)
        - t_new: 带有噪声的平移向量 (3x1 numpy array)
        """
        # 平移误差：对每个平移量加入独立的高斯噪声
        t_noise = np.random.normal(0, sigma_trans, size=(3,))
        # 均匀分布
        t_noise = np.random.uniform(-1, 1, size=(3,))*sigma_trans
        t_new = t_original + t_noise
        
        # 旋转误差：生成一个随机旋转轴和旋转角度
        theta_noise = np.random.normal(0, sigma_rot)  # 旋转角度的噪声
        # 均匀分布
        theta_noise = np.random.uniform(-1, 1)*sigma_rot  # 旋转角度的噪声
        random_axis = np.random.randn(3)  # 随机旋转轴
        random_axis /= np.linalg.norm(random_axis)  # 归一化旋转轴
        
        # 使用旋转轴和角度生成旋转矩阵噪声
        R_noise = Rotation.from_rotvec(theta_noise * random_axis).as_matrix()
        
        # 生成新的旋转矩阵
        R_new = R_original @ R_noise
        
        return R_new, R_noise, t_new, t_noise

    def __getitem__(self, index):
        anno = self.annos[index]  
        data = {}
        # Read image
        img_file = anno['img_file']
        pose_file = anno["pose_file"]
        intrin_file = anno["intrin_file"]
        gs_model = anno["gs_model"]
        
        image =  cv2.imread(img_file)
        pose_label = np.loadtxt(pose_file)
        Rot_label = pose_label[:3, :3]
        trans_label = pose_label[3:, 3]
        intrin_label = np.loadtxt(intrin_file)
        
        # Generate initial pose
        Rot_init, Rot_noise_gt, trans_init, trans_noise_gt = self.add_noise_to_pose(Rot_label, trans_label, self.sigma_rot, self.sigma_trans)
        
        data.update(
            {
                "Rot_noise_gt": torch.from_numpy(Rot_noise_gt).to(torch.float32),
                "trans_noise_gt": torch.from_numpy(trans_noise_gt).to(torch.float32),
            }
        )
        


    def __len__(self):
        return len(self.annos)


if __name__ == "__main__":
    anno_file = 'output/merged_anno/onepose_train_align/train.json'
    dataset = OnePoseDataset(anno_file)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=1)
    data = iter(data_loader)._next_data()
    pass