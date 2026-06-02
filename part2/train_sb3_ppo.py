import argparse

import gymnasium as gym
import panda_gym  # type: ignore[import-not-found]
from rand_wrapper import RandomizationWrapper
from wrappers import RewardShapingWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on PandaPush-v3")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="learning rate of the optimizer"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2048
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of training timesteps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment and model",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Select if you want to save or not the model weights"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments (>1 uses SubprocVecEnv for multi-core)"
    )
    return parser.parse_args()


def make_env(env_type: str, sampling_strategy: str, seed: int, rank: int):
    def _init():
        env = gym.make(
            "PandaPush-v3",
            render_mode="rgb_array",
            max_episode_steps=500,
            type=env_type,
            reward_type="dense",
        )
        env = RandomizationWrapper(env, mass_range=(1.0, 5.0), mode=sampling_strategy)
        env = RewardShapingWrapper(env, bonus_distance=0.05, bonus=100.0, time_penalty=1e-2)
        env = Monitor(env)

        env.reset(seed=seed + rank)
        return env
    return _init


def main() -> None:
    args = parse_args()

    env_fns = [make_env(args.env_type, args.sampling_strategy, args.seed, i) for i in range(args.n_envs)]
    if args.n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    #policy_kwargs = dict(net_arch=[256, 256])
    policy_kwargs = dict(net_arch=[512, 256, 128])
    
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log="./ppo_logs/",
        seed=args.seed,
        policy_kwargs=policy_kwargs
    )
    

    model.learn(total_timesteps=args.timesteps, log_interval=1)
    if args.save:
        save_name = f"ppo_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
        model.save(save_name)


if __name__ == "__main__":
    main()
