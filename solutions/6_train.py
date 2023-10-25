import os
import argparse
from typing import Optional
from distutils.util import strtobool
import numpy as np
import random
import time
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from utils import evaluate, make_incremental_dir


# parameters
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
            help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--env-id", type=str, default="Hopper-v4",
            help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(1e6),
            help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=9.80828e-05,
            help="the learning rate of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.99,
            help="the discount factor gamma")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")

    parser.add_argument("--num-steps", type=int, default=512,
        help="the number of steps to run in each environment per policy rollout")

    parser.add_argument("--num-minibatches", type=int, default=16,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=5,
            help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
            help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
            help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.00229519,
            help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.835671,
            help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.7,
            help="the maximum norm for the gradient clipping")

    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gae-lambda", type=float, default=0.99,
        help="the lambda for the general advantage estimation")

    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--visualize", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether visualize the agent performance")

    parser.add_argument("--load-checkpoint", type=str, default=None,
        help="whether load a model from a given checkpoint")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


# env creation function
def make_env(env_id, render_mode: Optional[str] = None):
    def thunk():
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


# agent
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=0.135),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.135),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def load_checkpoint(self, model_path, device):
        self.load_state_dict(torch.load(model_path, map_location=device))


if __name__ == "__main__":
    args = parse_args()

    run_name = f"{args.env_id}"
    runs_dir = f"runs/{run_name}"
    runs_dir = make_incremental_dir(runs_dir)

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

    # agent setup
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # use checkpoint
    if args.load_checkpoint:
        agent.load_checkpoint(args.load_checkpoint, device)
        print(f"model laoded from {args.load_checkpoint}")

    # rollout storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)  # type: ignore
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # start game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # rollout phase
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # action Logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # execute game
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            # log Data
            # only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # skip the envs that are not done
                if info is None:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

        # advantage estimation (A = rt + gamma * vt+1 - vt)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # learning phase
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)  # type: ignore
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # normalize Advantage
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policies ratio
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # entropy Loss
                entropy_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if update % (num_updates // 4) == 0 or update == num_updates:
            print("--------------------------------")
            model_dir = f"{runs_dir}/{update}"
            model_path = os.path.join(model_dir, "agent.pt")  # type: ignore

            if args.save_model:
                if not os.path.isdir(model_dir):
                    os.makedirs(model_dir)

                torch.save(agent.state_dict(), model_path)
                print(f"model saved to {model_path}")

                episodic_returns = evaluate(
                    model_path,
                    make_env,
                    args.env_id,
                    eval_episodes=10,
                    Model=Agent,  # type: ignore
                    device=device,
                )

                print(f"eval_reward= {episodic_returns.mean()}  +/- {episodic_returns.std()}")

                if args.visualize:
                    evaluate(
                        model_path,
                        make_env,
                        args.env_id,
                        eval_episodes=2,
                        Model=Agent,  # type: ignore
                        device=device,
                        render_mode="human",
                        print_rewards=True,
                    )

            print("--------------------------------")

    envs.close()
