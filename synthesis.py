import argparse
import numpy as np
import torch
import json

from hw_tts.model_handler import TTSGenerator

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config_path, weights):
    with open(config_path) as config_file:
        config = json.load(config_file)
    generator = TTSGenerator(config, checkpoint_path=weights)
    generator.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    # TODO: add resume option (with wandb)
    parser.add_argument(
        "-w",
        "--weights",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args = parser.parse_args()
    main(args.config, args.weights)