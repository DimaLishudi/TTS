import argparse
import numpy as np
import torch

from hw_tts.model_handler import TTSTrainer

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config, resume):
    trainer = TTSTrainer(config=config, checkpoint_path=resume)
    trainer.train_loop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    # TODO: add wandb resume option 
    # parser.add_arguments(
    #     "-i",
    #     "--wandb_id",
    #     default=None,
    #     type=int,
    #     help="wandb run id to resume (default: new run)"
    # )
    args = parser.parse_args()
    main(args.config, args.resume)