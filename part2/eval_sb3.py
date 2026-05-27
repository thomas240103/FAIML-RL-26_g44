import argparse
import os
import time

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401 - required so Panda envs are registered
from stable_baselines3 import DDPG, PPO, SAC

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "sac_push_none_source_500k.zip",
)


ALGO_LOADERS = {
    "sac": SAC,
    "ppo": PPO,
}


def infer_algorithm(model_path: str) -> str:
    model_name = os.path.basename(model_path).lower()

    for algo_name in ALGO_LOADERS:
        if model_name.startswith(f"{algo_name}_"):
            return algo_name

    raise ValueError(
        "Could not infer algorithm from model filename. "
        "Use a name like 'sac_...', 'ppo_...', or pass --algo explicitly."
    )


def load_model(model_path: str, algo: str, env):
    algo_name = algo.lower()
    if algo_name not in ALGO_LOADERS:
        supported_algos = ", ".join(sorted(ALGO_LOADERS))
        raise ValueError(f"Unsupported algorithm '{algo}'. Supported: {supported_algos}.")

    model_class = ALGO_LOADERS[algo_name]
    return model_class.load(model_path, env=env)


def evaluate(
    model_path: str,
    algo: str | None,
    n_episodes: int,
    deterministic: bool,
    render: bool,
    env_type: str,
    render_delay: float = 0.0,
) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model with model.save(...)."
        )

    resolved_algo = algo.lower() if algo is not None else infer_algorithm(model_path)
    render_mode = "human" if render else "rgb_array"
    env = gym.make("PandaPush-v3", render_mode=render_mode, type=env_type, reward_type="dense")
    model = load_model(model_path, resolved_algo, env=env)

    episode_returns = []
    successes = []

    print(
        f"Evaluating {resolved_algo.upper()} model '{os.path.basename(model_path)}' "
        f"on env_type='{env_type}' for {n_episodes} episodes."
    )

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            if render and render_delay > 0.0:
                time.sleep(render_delay)

        episode_returns.append(episode_return)

        if isinstance(info, dict) and "is_success" in info:
            successes.append(float(info["is_success"]))

        print(f"Episode {episode:03d} | return = {episode_return:.3f}")

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    print("\n=== Evaluation summary ===")
    print(f"Algorithm:   {resolved_algo.upper()}")
    print(f"Episodes:    {n_episodes}")
    print(f"Mean return: {returns.mean():.3f}")
    print(f"Std return:  {returns.std():.3f}")
    print(f"Min return:  {returns.min():.3f}")
    print(f"Max return:  {returns.max():.3f}")

    if successes:
        success_rate = float(np.mean(successes))
        print(f"Success rate: {success_rate:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an SB3 model on PandaPush-v3")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to an SB3 model zip file",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=sorted(ALGO_LOADERS),
        help="Algorithm used to train the model. If omitted, inferred from the filename.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of eval episodes",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render with a window (render_mode='human')",
    )
    parser.add_argument(
        "--render-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between steps when rendering (e.g. 0.05 for ~20 FPS)",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="target",
        choices=["source", "target"],
        help="Type of environment to evaluate on (default: target)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model_path=args.model_path,
        algo=args.algo,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        env_type=args.env_type,
        render_delay=args.render_delay,
    )
