import torch
import argparse
from distutils.util import strtobool
from utils import evaluate
train = __import__("solutions.6_train", fromlist=[None])  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser()

    # Algorithm specific arguments
    parser.add_argument("--model-path", type=str,
        help="the id of the environment")
    parser.add_argument("--env-id", type=str, default="Hopper-v4",
        help="the id of the environment")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--visualize", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether visualize the agent performance")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episodic_returns = evaluate(
        args.model_path,
        train.make_env,
        args.env_id,
        eval_episodes=10,
        Model=train.Agent,  # type: ignore
        device=device,
    )

    print(f"eval_reward= {episodic_returns.mean()}  +/- {episodic_returns.std()}")

    if args.visualize:
        evaluate(
            args.model_path,
            train.make_env,
            args.env_id,
            eval_episodes=5,
            Model=train.Agent,  # type: ignore
            device=device,
            render_mode="human",
            print_rewards=True,
        )
