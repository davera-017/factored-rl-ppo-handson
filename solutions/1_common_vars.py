import argparse
from distutils.util import strtobool
import numpy as np
import torch
import random
import time


# Parameters
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
            help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
            help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(1e6),
            help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
            help="the learning rate of the optimizer")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    run_name = f"{args.env_id}__{args.seed}__{int(time.time())}"

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print(args)
