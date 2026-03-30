import argparse
from match import test_match_model
from align import test_refine_model, test_joint_model
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser(description='Testing Gaussian Pose.')
    parser.add_argument("--testing_type", type=str, help="match | align | joint", default="match")
    parser.add_argument("--config", type=str, help="config file")

    args = parser.parse_args()

    if args.config is not None:
        cfg = OmegaConf.load(args.config)

    if args.testing_type == "match":
        test_match_model(cfg)
    elif args.testing_type == "align":
        test_refine_model(cfg)
    elif args.testing_type == "joint":
        test_joint_model(cfg)

if __name__ == '__main__':
    main()