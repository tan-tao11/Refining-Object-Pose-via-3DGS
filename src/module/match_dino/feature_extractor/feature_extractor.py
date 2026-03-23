import torch
import cv2
import torch.nn as nn

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

class feature_extractor_dino(nn.Module):
    def __init__(self, modelname = 'dino_vits16'):
        super().__init__()

        # Create feature extractor
        if "dinov2" in modelname:
            self._net = torch.hub.load("facebookresearch/dinov2", modelname)
            self._output_dim = self._net.norm.weight.shape[0] 
        elif "dino" in modelname:
            self._net = torch.hub.load("facebookresearch/dino:main", modelname)
            self._output_dim = self._net.norm.weight.shape[0]
        else:
            raise ValueError(f"Unknown model name {modelname}")
        
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)
    
    def _resnet_normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self._resnet_mean) / self._resnet_std
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img = self._resnet_normalize_image(img)
        x = self.extract_feature(img)

        return x

    def extract_feature(self, img: torch.Tensor) -> torch.Tensor:
        x = self._net.forward_features(img)["x_norm_patchtokens"]
        return x
    

if __name__ == "__main__":
    model_name = 'dinov2_vits14'

    image_file = '/data/tantao/my_methods/gs_pose/dataset_local/OnePose/train_data/0410-huiyuan-box/3DGS/images/0.png'
    img = cv2.imread(image_file)  # BGR 格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
    img = cv2.resize(img, (224, 224))
    img = torch.from_numpy(img).float() / 255.0  # 归一化
    img = img.permute(2, 0, 1).unsqueeze(0)  # 调整为 (1, C, H, W)

    extractor = feature_extractor_dino(model_name)
    x = extractor(img)

    print(x.shape)
