import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("part1/.matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("part1/.cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    {
        "label": "REINFORCE",
        "prefix": "reinforce_no_baseline",
        "color": "tab:blue",
    },
    {
        "label": "REINFORCE + baseline",
        "prefix": "reinforce_baseline_20",
        "color": "tab:orange",
    },
    {
        "label": "Actor-Critic",
        "prefix": "actor_critic",
        "color": "tab:green",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Part 1 mean/std comparison figures.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 99], help="Seeds to aggregate.")
    parser.add_argument("--results-dir", type=Path, default=Path("part1/results"), help="Directory containing .npz run files.")
    parser.add_argument("--plots-dir", type=Path, default=Path("part1/plots"), help="Directory for output plots.")
    parser.add_argument("--window", type=int, default=100, help="Moving-average window for training returns.")
    return parser.parse_args()


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid")


def load_runs(results_dir, prefix, seeds):
    runs = []
    for seed in seeds:
        path = results_dir / f"{prefix}_seed_{seed}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing result file: {path}")
        runs.append(np.load(path))
    return runs


def stack_equal_length(series):
    min_len = min(len(values) for values in series)
    return np.stack([values[:min_len] for values in series], axis=0)


def plot_train_comparison(args):
    plt.figure(figsize=(10, 6))

    for method in METHODS:
        runs = load_runs(args.results_dir, method["prefix"], args.seeds)
        smoothed = [moving_average(run["train_returns"], args.window) for run in runs]
        values = stack_equal_length(smoothed)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        start_episode = args.window if values.shape[1] != len(runs[0]["train_returns"]) else 1
        episodes = np.arange(start_episode, start_episode + len(mean))

        plt.plot(episodes, mean, label=method["label"], color=method["color"], linewidth=2)
        plt.fill_between(episodes, mean - std, mean + std, color=method["color"], alpha=0.18)

    plt.xlabel("Episode")
    plt.ylabel(f"Cumulative reward, {args.window}-episode moving average")
    plt.title("Part 1 training comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.plots_dir / "comparison_train_mean_std.png", dpi=150)
    plt.close()


def plot_eval_comparison(args):
    plt.figure(figsize=(10, 6))

    for method in METHODS:
        runs = load_runs(args.results_dir, method["prefix"], args.seeds)
        eval_returns = stack_equal_length([run["eval_returns"] for run in runs])
        eval_episodes = stack_equal_length([run["eval_episodes"] for run in runs])
        episodes = eval_episodes[0]
        mean = eval_returns.mean(axis=0)
        std = eval_returns.std(axis=0)

        plt.plot(episodes, mean, label=method["label"], color=method["color"], linewidth=2, marker="o", markersize=4)
        plt.fill_between(episodes, mean - std, mean + std, color=method["color"], alpha=0.18)

    plt.xlabel("Episode")
    plt.ylabel("Mean evaluation return")
    plt.title("Part 1 evaluation comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.plots_dir / "comparison_eval_mean_std.png", dpi=150)
    plt.close()


def main():
    args = parse_args()
    plot_train_comparison(args)
    plot_eval_comparison(args)
    print(f"Saved {args.plots_dir / 'comparison_train_mean_std.png'}")
    print(f"Saved {args.plots_dir / 'comparison_eval_mean_std.png'}")


if __name__ == "__main__":
    main()
