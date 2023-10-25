from typing import Callable, Optional
import numpy as np
import gymnasium as gym
import torch
import os


def make_incremental_dir(dir, i: int = 0):
    not_created = True
    i = 0
    while not_created:
        if os.path.isdir(dir + f"_{i}"):
            i += 1
            continue

        else:
            os.makedirs(dir + f"_{i}")
            return dir + f"_{i}"


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    render_mode: Optional[str] = None,
    print_rewards: bool = False
):
    __envs = gym.vector.SyncVectorEnv([make_env(env_id, render_mode)])
    agent = Model(__envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = __envs.reset()
    episodic_returns = []

    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        obs, _, _, _, infos = __envs.step(actions.cpu().numpy())

        if "final_info" not in infos:
            continue

        for info in infos["final_info"]:
            # skip the envs that are not done
            if info is None:
                continue
            if print_rewards:
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")

            episodic_returns += [info["episode"]["r"]]

    __envs.close()

    return np.array(episodic_returns)
