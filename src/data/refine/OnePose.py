import torch
import json
import cv2
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation


class OnePoseRefineDataset(Dataset):
    def __init__(
        self,
        anno_file,
        img_resize=(512, 512),
        sigma_rot=0.1,
        sigma_trans=0.05,
    ):
        super().__init__()

        self.img_resize = img_resize
        self.sigma_rot = sigma_rot
        self.sigma_trans = sigma_trans

        with open(anno_file, 'r') as f:
            self.annos = json.load(f)

    def add_noise_to_pose(self, R_gt, t_gt):
        """Add random noise to the ground-truth pose to simulate an initial
        (inaccurate) pose estimate."""
        t_noise = np.random.uniform(-1, 1, size=(3,)) * self.sigma_trans
        t_init = t_gt + t_noise

        theta_noise = np.random.uniform(-1, 1) * self.sigma_rot
        random_axis = np.random.randn(3)
        random_axis /= np.linalg.norm(random_axis) + 1e-8
        R_noise = Rotation.from_rotvec(theta_noise * random_axis).as_matrix()
        R_init = R_gt @ R_noise

        return R_init.astype(np.float32), t_init.astype(np.float32)

    def __getitem__(self, index):
        anno = self.annos[index]

        img_file = anno['img_file']
        pose_file = anno['pose_file']
        intrin_file = anno['intrin_file']
        gs_model = anno['gs_model']

        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_file}")
        image = cv2.resize(image, self.img_resize).astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)

        pose_gt = np.loadtxt(pose_file).astype(np.float32)   # 4x4
        R_gt = pose_gt[:3, :3]
        t_gt = pose_gt[:3, 3]
        K = np.loadtxt(intrin_file).astype(np.float32)        # 3x3

        R_init, t_init = self.add_noise_to_pose(R_gt, t_gt)

        return {
            'query_image': image,                                        # (1, H, W)
            'R_gt': torch.from_numpy(R_gt),                              # (3, 3)
            't_gt': torch.from_numpy(t_gt),                              # (3,)
            'initial_R': torch.from_numpy(R_init),                       # (3, 3)
            'initial_t': torch.from_numpy(t_init),                       # (3,)
            'K': torch.from_numpy(K),                                    # (3, 3)
            'gs_model': gs_model,
            'img_wh': list(self.img_resize),
        }

    def __len__(self):
        return len(self.annos)
