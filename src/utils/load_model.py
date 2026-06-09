from src.module.match.backbone import ResNetFPN_8_2


def load_backbone(cfg):
    if cfg.type == 'ResNetFPN':
        if cfg.resolution == [8, 2]:
            return ResNetFPN_8_2(cfg.resnetfpn)
        else:
            raise NotImplementedError
    else:
        raise ValueError("Please specify a valid backbone type.")