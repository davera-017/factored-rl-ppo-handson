import argparse
from distutils.util import strtobool
import numpy as np
import random
import time
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


# parameters
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
    parser.add_argument("--gamma", type=float, default=0.99,
            help="the discount factor gamma")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--gamma", type=float, default=0.99,
            help="the discount factor gamma")

    args = parser.parse_args()
    return args


# env creation function
def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

# agent
# TODO: orthogonal layer init function


# TODO: agent class
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # TODO: critic network

        # TODO: actor network
        pass

    # TODO: get value function
    def get_value(self, x):
        pass

    # TODO: get action andvalue function
    def get_action_and_value(self, x, action=None):
        pass


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

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id) for _ in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    print(f"obs space: {envs.single_observation_space.shape} -- act Space: {envs.single_action_space.shape}")
    print("----------------------------------------------------------------")

    # agent setup
    # TODO: initialize agent and optimizer

    # play
    # TODO: get action and values from observation
