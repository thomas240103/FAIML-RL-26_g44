import argparse
import os
import time

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401 - required so Panda envs are registered
from stable_baselines3 import PPO, SAC

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "sac_push_none_source_500k.zip",
)


ALGO_LOADERS = {
    "sac": SAC,
    "ppo": PPO,
}

MASS_MODES = ("fixed", "uniform", "grid")
FRICTION_MODES = ("fixed", "uniform")


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


def parse_mass_values(raw_values: str | None) -> list[float] | None:
    if raw_values is None:
        return None

    values = []
    for raw_value in raw_values.split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        values.append(float(raw_value))

    if not values:
        raise ValueError("--mass-values must contain at least one numeric value.")

    if any(value <= 0.0 for value in values):
        raise ValueError("Mass values must be positive.")

    return values


def get_nominal_mass(env) -> float:
    return float(env.unwrapped.task.current_mass)


def set_object_mass(env, mass: float) -> None:
    task = env.unwrapped.task
    sim = task.sim
    object_body_id = sim._bodies_idx["object"]

    sim.physics_client.changeDynamics(
        bodyUniqueId=object_body_id,
        linkIndex=-1,
        mass=float(mass),
    )
    task.current_mass = float(mass)


def get_dynamics_info(env, body_name: str) -> tuple:
    sim = env.unwrapped.task.sim
    body_id = sim._bodies_idx[body_name]
    return sim.physics_client.getDynamicsInfo(body_id, -1)


def get_nominal_friction(env) -> dict[str, float]:
    object_info = get_dynamics_info(env, "object")
    table_info = get_dynamics_info(env, "table")
    return {
        "object_lateral": float(object_info[1]),
        "table_lateral": float(table_info[1]),
        "object_spinning": float(object_info[7]),
    }


def set_friction(env, values: dict[str, float]) -> None:
    sim = env.unwrapped.task.sim
    sim.set_lateral_friction("object", -1, values["object_lateral"])
    sim.set_lateral_friction("table", -1, values["table_lateral"])
    sim.set_spinning_friction("object", -1, values["object_spinning"])


def validate_nonnegative_value(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")


def validate_nonnegative_range(name: str, low: float, high: float) -> None:
    if low < 0.0 or high < 0.0 or low > high:
        raise ValueError(f"{name} range must be non-negative and ordered.")


def build_mass_schedule(
    mode: str,
    n_episodes: int,
    nominal_mass: float,
    fixed_mass: float | None,
    mass_min: float,
    mass_max: float,
    mass_values: list[float] | None,
    mass_grid_size: int,
    rng: np.random.Generator,
) -> list[float]:
    if n_episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")

    if mode not in MASS_MODES:
        supported_modes = ", ".join(MASS_MODES)
        raise ValueError(f"Unsupported eval mass mode '{mode}'. Supported: {supported_modes}.")

    if mass_min <= 0.0 or mass_max <= 0.0 or mass_min > mass_max:
        raise ValueError("--mass-min and --mass-max must be positive and ordered.")

    if mode == "fixed":
        mass = nominal_mass if fixed_mass is None else fixed_mass
        if mass <= 0.0:
            raise ValueError("--fixed-mass must be positive.")
        return [float(mass)] * n_episodes

    if mode == "uniform":
        return rng.uniform(mass_min, mass_max, size=n_episodes).astype(float).tolist()

    if mass_values is None:
        if mass_grid_size <= 0:
            raise ValueError("--mass-grid-size must be greater than zero.")
        mass_values = np.linspace(mass_min, mass_max, num=mass_grid_size).astype(float).tolist()

    schedule = []
    for episode_idx in range(n_episodes):
        schedule.append(float(mass_values[episode_idx % len(mass_values)]))
    return schedule


def build_friction_schedule(
    mode: str,
    n_episodes: int,
    nominal_friction: dict[str, float],
    object_lateral_friction: float | None,
    table_lateral_friction: float | None,
    object_spinning_friction: float | None,
    object_lateral_friction_min: float,
    object_lateral_friction_max: float,
    table_lateral_friction_min: float,
    table_lateral_friction_max: float,
    object_spinning_friction_min: float,
    object_spinning_friction_max: float,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    if mode not in FRICTION_MODES:
        supported_modes = ", ".join(FRICTION_MODES)
        raise ValueError(f"Unsupported eval friction mode '{mode}'. Supported: {supported_modes}.")

    validate_nonnegative_range(
        "object lateral friction",
        object_lateral_friction_min,
        object_lateral_friction_max,
    )
    validate_nonnegative_range(
        "table lateral friction",
        table_lateral_friction_min,
        table_lateral_friction_max,
    )
    validate_nonnegative_range(
        "object spinning friction",
        object_spinning_friction_min,
        object_spinning_friction_max,
    )

    if mode == "fixed":
        friction = {
            "object_lateral": (
                nominal_friction["object_lateral"]
                if object_lateral_friction is None
                else float(object_lateral_friction)
            ),
            "table_lateral": (
                nominal_friction["table_lateral"]
                if table_lateral_friction is None
                else float(table_lateral_friction)
            ),
            "object_spinning": (
                nominal_friction["object_spinning"]
                if object_spinning_friction is None
                else float(object_spinning_friction)
            ),
        }
        for name, value in friction.items():
            validate_nonnegative_value(name, value)
        return [friction.copy() for _ in range(n_episodes)]

    schedule = []
    for _ in range(n_episodes):
        schedule.append(
            {
                "object_lateral": float(
                    rng.uniform(object_lateral_friction_min, object_lateral_friction_max)
                ),
                "table_lateral": float(
                    rng.uniform(table_lateral_friction_min, table_lateral_friction_max)
                ),
                "object_spinning": float(
                    rng.uniform(object_spinning_friction_min, object_spinning_friction_max)
                ),
            }
        )
    return schedule


def print_bin_summary(
    values: np.ndarray,
    returns: np.ndarray,
    successes: np.ndarray | None,
    bin_count: int,
    value_name: str,
    unit: str = "",
) -> None:
    if len(values) == 0 or np.allclose(values.min(), values.max()):
        return

    bin_count = max(1, min(bin_count, len(values)))
    edges = np.linspace(float(values.min()), float(values.max()), num=bin_count + 1)
    unit_suffix = f" {unit}" if unit else ""

    print(f"\n=== {value_name}-bin summary ===")
    for idx in range(bin_count):
        low = edges[idx]
        high = edges[idx + 1]
        if idx == bin_count - 1:
            mask = (values >= low) & (values <= high)
            label = f"[{low:.2f}, {high:.2f}]"
        else:
            mask = (values >= low) & (values < high)
            label = f"[{low:.2f}, {high:.2f})"

        count = int(mask.sum())
        if count == 0:
            continue

        summary = (
            f"{value_name} {label}{unit_suffix} | episodes={count:3d} "
            f"| mean return={returns[mask].mean():8.3f} "
            f"| min return={returns[mask].min():8.3f}"
        )
        if successes is not None:
            summary += f" | success rate={successes[mask].mean():6.2%}"
        print(summary)


def print_mass_summary(
    episode_masses: list[float],
    returns: np.ndarray,
    successes: list[float],
    heavy_mass_threshold: float,
    mass_bins: int,
) -> None:
    masses = np.array(episode_masses, dtype=np.float32)
    success_array = None
    if len(successes) == len(returns):
        success_array = np.array(successes, dtype=np.float32)

    print("\n=== Mass summary ===")
    print(f"Mass min:    {masses.min():.3f} kg")
    print(f"Mass mean:   {masses.mean():.3f} kg")
    print(f"Mass max:    {masses.max():.3f} kg")

    if success_array is not None:
        heavy_mask = masses >= heavy_mass_threshold
        if heavy_mask.any():
            print(
                f"Success rate >= {heavy_mass_threshold:.2f} kg: "
                f"{success_array[heavy_mask].mean():.2%} ({int(heavy_mask.sum())} episodes)"
            )

    print_bin_summary(masses, returns, success_array, mass_bins, "Mass", "kg")


def print_friction_summary(
    episode_frictions: list[dict[str, float]],
    returns: np.ndarray,
    successes: list[float],
    friction_bins: int,
) -> None:
    if not episode_frictions:
        return

    success_array = None
    if len(successes) == len(returns):
        success_array = np.array(successes, dtype=np.float32)

    labels = {
        "object_lateral": "Object lateral friction",
        "table_lateral": "Table lateral friction",
        "object_spinning": "Object spinning friction",
    }

    print("\n=== Friction summary ===")
    for key, label in labels.items():
        values = np.array([friction[key] for friction in episode_frictions], dtype=np.float32)
        print(
            f"{label}: min={values.min():.4f} "
            f"mean={values.mean():.4f} max={values.max():.4f}"
        )

    for key, label in labels.items():
        values = np.array([friction[key] for friction in episode_frictions], dtype=np.float32)
        print_bin_summary(values, returns, success_array, friction_bins, label)


def evaluate(
    model_path: str,
    algo: str | None,
    n_episodes: int,
    deterministic: bool,
    render: bool,
    env_type: str,
    eval_mass_mode: str,
    mass_min: float,
    mass_max: float,
    fixed_mass: float | None,
    mass_values: list[float] | None,
    mass_grid_size: int,
    max_episode_steps: int,
    seed: int | None,
    mass_bins: int,
    heavy_mass_threshold: float,
    eval_friction_mode: str,
    object_lateral_friction: float | None,
    table_lateral_friction: float | None,
    object_spinning_friction: float | None,
    object_lateral_friction_min: float,
    object_lateral_friction_max: float,
    table_lateral_friction_min: float,
    table_lateral_friction_max: float,
    object_spinning_friction_min: float,
    object_spinning_friction_max: float,
    friction_bins: int,
    render_delay: float = 0.0,
) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model with model.save(...)."
        )
    if max_episode_steps <= 0:
        raise ValueError("--max-episode-steps must be greater than zero.")
    if heavy_mass_threshold <= 0.0:
        raise ValueError("--heavy-mass-threshold must be positive.")
    if mass_bins <= 0:
        raise ValueError("--mass-bins must be greater than zero.")
    if friction_bins <= 0:
        raise ValueError("--friction-bins must be greater than zero.")

    resolved_algo = algo.lower() if algo is not None else infer_algorithm(model_path)
    render_mode = "human" if render else "rgb_array"
    env = gym.make(
        "PandaPush-v3",
        render_mode=render_mode,
        type=env_type,
        reward_type="dense",
        max_episode_steps=max_episode_steps,
    )
    rng = np.random.default_rng(seed)
    mass_schedule = build_mass_schedule(
        mode=eval_mass_mode,
        n_episodes=n_episodes,
        nominal_mass=get_nominal_mass(env),
        fixed_mass=fixed_mass,
        mass_min=mass_min,
        mass_max=mass_max,
        mass_values=mass_values,
        mass_grid_size=mass_grid_size,
        rng=rng,
    )
    friction_schedule = build_friction_schedule(
        mode=eval_friction_mode,
        n_episodes=n_episodes,
        nominal_friction=get_nominal_friction(env),
        object_lateral_friction=object_lateral_friction,
        table_lateral_friction=table_lateral_friction,
        object_spinning_friction=object_spinning_friction,
        object_lateral_friction_min=object_lateral_friction_min,
        object_lateral_friction_max=object_lateral_friction_max,
        table_lateral_friction_min=table_lateral_friction_min,
        table_lateral_friction_max=table_lateral_friction_max,
        object_spinning_friction_min=object_spinning_friction_min,
        object_spinning_friction_max=object_spinning_friction_max,
        rng=rng,
    )
    model = load_model(model_path, resolved_algo, env=env)

    episode_returns = []
    successes = []
    episode_masses = []
    episode_frictions = []

    print(
        f"Evaluating {resolved_algo.upper()} model '{os.path.basename(model_path)}' "
        f"on env_type='{env_type}' for {n_episodes} episodes."
    )
    schedule_min = min(mass_schedule)
    schedule_max = max(mass_schedule)
    print(
        f"Mass mode: {eval_mass_mode} | sampled range=[{schedule_min:.2f}, {schedule_max:.2f}] kg "
        f"| horizon={max_episode_steps} steps | seed={seed}"
    )
    friction_ranges = {
        key: (
            min(friction[key] for friction in friction_schedule),
            max(friction[key] for friction in friction_schedule),
        )
        for key in ("object_lateral", "table_lateral", "object_spinning")
    }
    print(
        "Friction mode: "
        f"{eval_friction_mode} | object_lateral=[{friction_ranges['object_lateral'][0]:.4f}, "
        f"{friction_ranges['object_lateral'][1]:.4f}] | table_lateral=["
        f"{friction_ranges['table_lateral'][0]:.4f}, {friction_ranges['table_lateral'][1]:.4f}] "
        f"| object_spinning=[{friction_ranges['object_spinning'][0]:.4f}, "
        f"{friction_ranges['object_spinning'][1]:.4f}]"
    )

    for episode, (episode_mass, episode_friction) in enumerate(
        zip(mass_schedule, friction_schedule),
        start=1,
    ):
        set_object_mass(env, episode_mass)
        set_friction(env, episode_friction)
        reset_kwargs = {}
        if seed is not None:
            reset_kwargs["seed"] = seed + episode - 1

        obs, info = env.reset(**reset_kwargs)
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
        episode_masses.append(episode_mass)
        episode_frictions.append(episode_friction)

        final_success = None
        if isinstance(info, dict) and "is_success" in info:
            final_success = float(info["is_success"])
            successes.append(final_success)

        success_text = ""
        if final_success is not None:
            success_text = f" | success = {int(final_success)}"
        print(
            f"Episode {episode:03d} | mass = {episode_mass:.3f} kg "
            f"| obj_mu = {episode_friction['object_lateral']:.4f} "
            f"| table_mu = {episode_friction['table_lateral']:.4f} "
            f"| obj_spin = {episode_friction['object_spinning']:.4f} "
            f"| return = {episode_return:.3f}{success_text}"
        )

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

    print_mass_summary(
        episode_masses=episode_masses,
        returns=returns,
        successes=successes,
        heavy_mass_threshold=heavy_mass_threshold,
        mass_bins=mass_bins,
    )
    print_friction_summary(
        episode_frictions=episode_frictions,
        returns=returns,
        successes=successes,
        friction_bins=friction_bins,
    )


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
    parser.add_argument(
        "--eval-mass-mode",
        type=str,
        default="fixed",
        choices=MASS_MODES,
        help=(
            "How to choose object mass at eval time: fixed uses the env nominal mass "
            "unless --fixed-mass is set; uniform samples in [--mass-min, --mass-max]; "
            "grid cycles through --mass-values or an evenly spaced grid."
        ),
    )
    parser.add_argument(
        "--fixed-mass",
        type=float,
        default=None,
        help="Fixed eval mass in kg. Defaults to source/target nominal mass.",
    )
    parser.add_argument(
        "--mass-min",
        type=float,
        default=5.0,
        help="Lower mass bound in kg for uniform/grid eval.",
    )
    parser.add_argument(
        "--mass-max",
        type=float,
        default=10.0,
        help="Upper mass bound in kg for uniform/grid eval.",
    )
    parser.add_argument(
        "--mass-values",
        type=str,
        default=None,
        help="Comma-separated masses in kg for grid eval, e.g. '5,6,7,8,9,10'.",
    )
    parser.add_argument(
        "--mass-grid-size",
        type=int,
        default=6,
        help="Number of evenly spaced masses for grid eval when --mass-values is omitted.",
    )
    parser.add_argument(
        "--mass-bins",
        type=int,
        default=5,
        help="Number of bins used for mass-conditioned summary metrics.",
    )
    parser.add_argument(
        "--heavy-mass-threshold",
        type=float,
        default=8.0,
        help="Threshold in kg for reporting heavy-object success rate.",
    )
    parser.add_argument(
        "--eval-friction-mode",
        type=str,
        default="fixed",
        choices=FRICTION_MODES,
        help=(
            "How to choose friction at eval time: fixed uses nominal PyBullet values "
            "unless fixed friction flags are set; uniform samples each friction value "
            "from its configured range."
        ),
    )
    parser.add_argument(
        "--object-lateral-friction",
        type=float,
        default=None,
        help="Fixed object lateral friction. Defaults to the nominal PyBullet value.",
    )
    parser.add_argument(
        "--table-lateral-friction",
        type=float,
        default=None,
        help="Fixed table lateral friction. Defaults to the nominal PyBullet value.",
    )
    parser.add_argument(
        "--object-spinning-friction",
        type=float,
        default=None,
        help="Fixed object spinning friction. Defaults to the nominal PyBullet value.",
    )
    parser.add_argument(
        "--object-lateral-friction-min",
        type=float,
        default=0.2,
        help="Lower bound for uniform object lateral friction.",
    )
    parser.add_argument(
        "--object-lateral-friction-max",
        type=float,
        default=1.5,
        help="Upper bound for uniform object lateral friction.",
    )
    parser.add_argument(
        "--table-lateral-friction-min",
        type=float,
        default=0.2,
        help="Lower bound for uniform table lateral friction.",
    )
    parser.add_argument(
        "--table-lateral-friction-max",
        type=float,
        default=1.5,
        help="Upper bound for uniform table lateral friction.",
    )
    parser.add_argument(
        "--object-spinning-friction-min",
        type=float,
        default=0.0,
        help="Lower bound for uniform object spinning friction.",
    )
    parser.add_argument(
        "--object-spinning-friction-max",
        type=float,
        default=0.01,
        help="Upper bound for uniform object spinning friction.",
    )
    parser.add_argument(
        "--friction-bins",
        type=int,
        default=5,
        help="Number of bins used for friction-conditioned summary metrics.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=500,
        help="Explicit evaluation horizon.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional eval seed. If set, episode i uses seed + i.",
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
        eval_mass_mode=args.eval_mass_mode,
        mass_min=args.mass_min,
        mass_max=args.mass_max,
        fixed_mass=args.fixed_mass,
        mass_values=parse_mass_values(args.mass_values),
        mass_grid_size=args.mass_grid_size,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        mass_bins=args.mass_bins,
        heavy_mass_threshold=args.heavy_mass_threshold,
        eval_friction_mode=args.eval_friction_mode,
        object_lateral_friction=args.object_lateral_friction,
        table_lateral_friction=args.table_lateral_friction,
        object_spinning_friction=args.object_spinning_friction,
        object_lateral_friction_min=args.object_lateral_friction_min,
        object_lateral_friction_max=args.object_lateral_friction_max,
        table_lateral_friction_min=args.table_lateral_friction_min,
        table_lateral_friction_max=args.table_lateral_friction_max,
        object_spinning_friction_min=args.object_spinning_friction_min,
        object_spinning_friction_max=args.object_spinning_friction_max,
        friction_bins=args.friction_bins,
        render_delay=args.render_delay,
    )
