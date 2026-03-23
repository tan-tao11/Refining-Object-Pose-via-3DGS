import argparse
from match import test_match_model
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser(description='Training Gaussian Pose.')
    parser.add_argument("--testing_type", type=str, help="match | align", default="match")
    parser.add_argument("--config", type=str, help="config file")

    args = parser.parse_args()

    if args.config is not None:
        cfg = OmegaConf.load(args.config)

    if args.testing_type == "match":
        test_match_model(cfg)
    # elif args.testing_type == "align":
    #     train_match_model(cfg)

if __name__ == '__main__':
    main()