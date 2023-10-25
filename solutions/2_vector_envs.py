import argparse
from distutils.util import strtobool
import numpy as np
import torch
import random
import time
import gymnasium as gym


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
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")

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

    # single env
    env = gym.make(args.env_id)
    print(f"Obs space: {env.observation_space.shape} -- Act space: {env.action_space.shape}")

    obs, _ = env.reset()
    num_steps = 1000
    episodic_return = 0
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, infos = env.step(action)
        done = np.logical_or(truncated, terminated)

        episodic_return += reward

        if done:
            print(f"global_step={step}, episodic_return={episodic_return}")
            obs, _ = env.reset()
            episodic_return = 0

    # sync envs
    def make_env(env_id):
        def thunk():
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        return thunk

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id) for _ in range(args.num_envs)]
    )
    print(f"Sync Obs space: {envs.observation_space.shape} -- Sync Act Space: {envs.action_space.shape}")
    obs, _ = envs.reset()
    num_steps = 1000
    for step in range(num_steps):
        actions = envs.action_space.sample()
        obs, reward, terminated, truncated, infos = envs.step(actions)
        done = np.logical_or(truncated, terminated)

        if "final_info" not in infos:
            continue

        for idx, info in enumerate(infos["final_info"]):
            # Skip the envs that are not done
            if info is None:
                continue
            print(f"global_step={step}, env_id={idx}, episodic_return={info['episode']['r']}")
