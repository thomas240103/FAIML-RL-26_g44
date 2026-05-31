"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("part1/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("part1/.cache").resolve()))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser(description="Train REINFORCE or Actor-Critic on Hopper-v4")
    parser.add_argument(
        "--alg",
        choices=["reinforce", "actor-critic"],
        default="reinforce",
        help="Training algorithm",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=0.0,
        help="Constant baseline subtracted from REINFORCE discounted returns",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3000,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print training stats every N episodes",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="Evaluate every N episodes",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of episodes per evaluation",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render training episodes in a human window",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output name prefix for the learning curve",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot without opening an interactive window",
    )
    return parser.parse_args()


def evaluate_policy(agent, eval_env, n_eval_episodes=5):
    """Run deterministic evaluation episodes and return mean cumulative reward."""
    eval_returns = []

    for _ in range(n_eval_episodes):
        done = False
        state, _ = eval_env.reset()
        episode_return = 0.0

        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            action_np = action.detach().cpu().numpy()
            action_np = np.clip(action_np, eval_env.action_space.low, eval_env.action_space.high)
            next_state, reward, terminated, truncated, _ = eval_env.step(action_np)
            done = terminated or truncated
            episode_return += reward
            state = next_state

        eval_returns.append(episode_return)

    return float(np.mean(eval_returns))


def main():
    args = parse_args()
    render = args.render

    if render:
        env = gym.make('Hopper-v4', render_mode='human')
    else:
        env = gym.make('Hopper-v4', render_mode='rgb_array')

    eval_env = gym.make('Hopper-v4', render_mode='rgb_array')
    env.action_space.seed(args.seed)
    eval_env.action_space.seed(args.seed + 1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy=Policy(state_space=env.observation_space.shape[0], action_space=env.action_space.shape[0])
    agent = Agent(policy)


    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    n_episodes = args.episodes
    print_every = args.print_every
    eval_every = args.eval_every
    n_eval_episodes = args.eval_episodes

    print(f"Algorithm: {args.alg}")
    if args.alg == "reinforce":
        print(f"Baseline: {args.baseline}")
    print(f"Seed: {args.seed}")

    train_returns = []
    eval_episodes = []
    eval_returns = []

    for episode in range(n_episodes):
        done = False
        current_state, info = env.reset(seed=args.seed + episode)  # Reset environment to initial state
        episode_return = 0.0

        while not done:

            action, action_log_prob, vote = agent.get_action(current_state)
            action_np = action.detach().cpu().numpy()
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
            state, reward, terminated, truncated, _ = env.step(action_np)  # Step the simulator to the next timestep

            done = terminated or truncated
            episode_return += reward

            agent.store_outcome(state=current_state, next_state=state, reward=reward, action_log_prob=action_log_prob, done=done, state_value=vote)
            current_state = state

            if render:
                env.render()

        train_returns.append(episode_return)
        if (episode + 1) % print_every == 0:
            recent_mean = np.mean(train_returns[-print_every:])
            print(f"Episode {episode + 1}/{n_episodes} | Return: {episode_return:.2f} | Mean({print_every}): {recent_mean:.2f}")

        if (episode + 1) % eval_every == 0:
            eval_mean_return = evaluate_policy(agent, eval_env, n_eval_episodes=n_eval_episodes)
            eval_episodes.append(episode + 1)
            eval_returns.append(eval_mean_return)
            print(f"[EVAL] Episode {episode + 1} | Mean return over {n_eval_episodes} episodes: {eval_mean_return:.2f}")

        agent.update_policy(alg=args.alg, baseline=args.baseline)

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_returns, label='Train return per episode', alpha=0.4)

    window = 100
    if len(train_returns) >= window:
        moving_avg = np.convolve(train_returns, np.ones(window) / window, mode='valid')
        plt.plot(
            np.arange(window - 1, len(train_returns)),
            moving_avg,
            label=f'Train moving average ({window})',
            linewidth=2
        )

    if eval_returns:
        plt.plot(eval_episodes, eval_returns, marker='o', linestyle='-', label='Evaluation mean return')

    plt.xlabel('Episode')
    plt.ylabel('Cumulative reward')
    title = f"Learning curve - {args.alg}"
    if args.alg == "reinforce" and args.baseline != 0:
        title += f" baseline={args.baseline:g}"
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    run_name = args.out
    if run_name is None:
        baseline_tag = f"_baseline_{args.baseline:g}" if args.alg == "reinforce" and args.baseline != 0 else ""
        run_name = f"{args.alg}{baseline_tag}_seed_{args.seed}"

    plots_dir = Path("part1/plots")
    results_dir = Path("part1/results")
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        results_dir / f"{run_name}.npz",
        train_returns=np.array(train_returns, dtype=np.float32),
        eval_episodes=np.array(eval_episodes, dtype=np.int32),
        eval_returns=np.array(eval_returns, dtype=np.float32),
        algorithm=args.alg,
        baseline=np.array(args.baseline, dtype=np.float32),
        seed=np.array(args.seed, dtype=np.int32),
    )
    plt.savefig(plots_dir / f'{run_name}_learning_curve.png', dpi=150)
    if not args.no_show:
        plt.show()
    plt.close()

    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
