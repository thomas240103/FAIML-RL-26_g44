import argparse
import subprocess
import sys


EXPERIMENTS = [
#    {
#        "name": "reinforce_no_baseline",
#        "args": ["--alg", "reinforce"],
#    },
#    {
#        "name": "reinforce_baseline_20",
#        "args": ["--alg", "reinforce", "--baseline", "20"],
#    },
    {
        "name": "actor_critic",
        "args": ["--alg", "actor-critic"],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run all Part 1 Hopper experiments.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 99], help="Seeds to run.")
    parser.add_argument("--episodes", type=int, default=5000, help="Training episodes per run.")
    parser.add_argument("--eval-every", type=int, default=100, help="Evaluation frequency.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Episodes per evaluation.")
    parser.add_argument("--print-every", type=int, default=50, help="Training print frequency.")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots after training.")
    return parser.parse_args()


def main():
    args = parse_args()

    for seed in args.seeds:
        for experiment in EXPERIMENTS:
            run_name = f"{experiment['name']}_seed_{seed}"
            command = [
                sys.executable,
                "part1/train.py",
                *experiment["args"],
                "--episodes",
                str(args.episodes),
                "--seed",
                str(seed),
                "--eval-every",
                str(args.eval_every),
                "--eval-episodes",
                str(args.eval_episodes),
                "--print-every",
                str(args.print_every),
                "--out",
                run_name,
                "--no-show",
            ]

            print(f"\nRunning {run_name}")
            subprocess.run(command, check=True)

    if args.plot:
        subprocess.run([sys.executable, "part1/plot_comparison.py"], check=True)


if __name__ == "__main__":
    main()
