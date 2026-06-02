import argparse

import gymnasium as gym
import panda_gym  # type: ignore[import-not-found]

from rand_wrapper import RandomizationWrapper
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on PandaPush-v3")
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
        type=bool,
        default=False,
        choices=[True, False],
        help="Select if you want to save or not the model weights",
    )
    parser.add_argument(
        "--her",
        action="store_true",
        default=False,
        help="Enable Hindsight Experience Replay (HER)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for actor and critic",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Minibatch size for each gradient update",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="Gradient updates per env step (-1 = same as steps collected)",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=100,
        help="Steps before starting to train",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1_000_000,
        help="Size of the replay buffer",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Soft update coefficient for target networks",
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="Collect this many steps before each training cycle",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training (mps for Apple Silicon)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor of the rewards"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = gym.make(
        "PandaPush-v3",
        type=args.env_type,
        reward_type="dense",
    )

    env = RandomizationWrapper(
        env,
        mass_range=(1.0, 5.0),
        mode=args.sampling_strategy,
    )
    env.reset(seed=args.seed)

    replay_buffer_class = HerReplayBuffer if args.her else None
    replay_buffer_kwargs = dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ) if args.her else None

    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_steps=args.gradient_steps,
        learning_starts=args.learning_starts,
        buffer_size=args.buffer_size,
        tau=args.tau,
        train_freq=args.train_freq,
        ent_coef="auto",
        verbose=1,
        tensorboard_log="./sac_logs/",
        seed=args.seed,
        device=args.device,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
        gamma=args.gamma
    )
    model.learn(total_timesteps=args.timesteps, log_interval=4)
    if args.save:
        save_name = f"sac_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
        if args.her:
            save_name += "_her"
        model.save(save_name)


if __name__ == "__main__":
    main()
