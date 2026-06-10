# Part 1: Hopper Environment

This part focuses on implementing Reinforcement Learning algorithms from scratch and testing them on the continuous control MuJoCo `Hopper-v4` environment.

## Code Overview

- **`agent.py`**: Contains the PyTorch neural network architectures for the `Policy` (Actor and Critic networks) and the `Agent` class logic. It includes the `update_policy` method where the core mathematical updates for REINFORCE (with/without baseline) and Actor-Critic algorithms are implemented.
- **`train.py`**: The main script to train the algorithms. It sets up the environment, initializes the agent, runs the training loop, periodically evaluates the policy, and saves the learning curve plots in the `plots/` folder.
- **`test_random_policy.py`**: A helper script to interact with the environment using a random policy, useful for inspecting state/action spaces and the simulator's rendering.
- **`run_part1_experiments.py`**: An automated script to run multiple experiments sequentially.
- **`plot_comparison.py`**: Utility to compare learning curves across different seeds and models.

---

## Instructions

This document lists the commands for the Part 1 Hopper experiments requested in `FAIML_RL_2026.pdf`.

Run every command from the repository root.

## Environment

Create and activate a Python environment, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you already use Conda:

```bash
conda activate venv
python -m pip install -r requirements.txt
```

Create the output directory used by `part1/train.py`:

```bash
mkdir -p part1/plots
```

## Task 1 - Hopper Environment

Run the random policy script to inspect the environment visually and print the state and action spaces:

```bash
python part1/test_random_policy.py
```

For a non-rendering command that prints the state space, action space, MuJoCo body names, body masses, robot DoFs, per-body DoFs, and number of actuators:

```bash
python - <<'PY'
import gymnasium as gym

env = gym.make("Hopper-v4", render_mode="rgb_array")
env.reset(seed=0)
model = env.unwrapped.model
body_names = getattr(model, "body_names", None)
if body_names is None:
    try:
        body_names = env.unwrapped.model_names.body_names
    except AttributeError:
        import mujoco
        body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(model.nbody)
        ]

print("State space:", env.observation_space)
print("Action space:", env.action_space)
print("Body names:", body_names)
print("Body masses:", model.body_mass)
print("Number of DoFs:", model.nv)
print("DoFs per body:", model.body_dofnum)
print("Number of actuators:", model.nu)

env.close()
PY
```

Use these outputs to answer the Task 1 guiding questions about whether the state and action spaces are discrete or continuous, and what the Hopper link masses are.

## Task 2 - REINFORCE

### REINFORCE Without Baseline

```bash
time python part1/train.py \
  --alg reinforce \
  --episodes 3000 \
  --seed 0 \
  --eval-every 100 \
  --eval-episodes 5 \
  --out reinforce_no_baseline_seed_0 \
  --no-show
```

### REINFORCE With Constant Baseline

The assignment asks to compare REINFORCE without baseline against REINFORCE with a constant baseline. Start with the baseline value below, then repeat with other values to justify the chosen baseline in the report.

```bash
time python part1/train.py \
  --alg reinforce \
  --baseline 20 \
  --episodes 3000 \
  --seed 0 \
  --eval-every 100 \
  --eval-episodes 5 \
  --out reinforce_baseline_20_seed_0 \
  --no-show
```

Optional baseline sweep:

```bash
for baseline in 0 10 20 50 100; do
  time python part1/train.py \
    --alg reinforce \
    --baseline "$baseline" \
    --episodes 3000 \
    --seed 0 \
    --eval-every 100 \
    --eval-episodes 5 \
    --out "reinforce_baseline_${baseline}_seed_0" \
    --no-show
done
```

## Task 3 - Actor-Critic

```bash
time python part1/train.py \
  --alg actor-critic \
  --episodes 3000 \
  --seed 0 \
  --eval-every 100 \
  --eval-episodes 5 \
  --out actor_critic_seed_0 \
  --no-show
```

Compare this run against both REINFORCE variants in terms of reward, time consumption, learning stability, and convergence speed.

## Multi-Seed Runs

For more reliable plots and report numbers, repeat the three main training configurations with multiple seeds:

```bash
for seed in 0 1 2; do
  time python part1/train.py \
    --alg reinforce \
    --episodes 3000 \
    --seed "$seed" \
    --eval-every 100 \
    --eval-episodes 5 \
    --out "reinforce_no_baseline_seed_${seed}" \
    --no-show

  time python part1/train.py \
    --alg reinforce \
    --baseline 20 \
    --episodes 3000 \
    --seed "$seed" \
    --eval-every 100 \
    --eval-episodes 5 \
    --out "reinforce_baseline_20_seed_${seed}" \
    --no-show

  time python part1/train.py \
    --alg actor-critic \
    --episodes 3000 \
    --seed "$seed" \
    --eval-every 100 \
    --eval-episodes 5 \
    --out "actor_critic_seed_${seed}" \
    --no-show
done
```

## Output Files

Each training command saves a learning curve in:

```text
part1/plots/<out>_learning_curve.png
```

Examples:

```text
part1/plots/reinforce_no_baseline_seed_0_learning_curve.png
part1/plots/reinforce_baseline_20_seed_0_learning_curve.png
part1/plots/actor_critic_seed_0_learning_curve.png
```

The `time` prefix prints wall-clock runtime, which is useful for the report questions about time consumption.

## Useful CLI Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--alg` | `reinforce` | Training algorithm. Choices: `reinforce`, `actor-critic`. |
| `--baseline` | `0.0` | Constant baseline subtracted from REINFORCE discounted returns. |
| `--episodes` | `3000` | Number of training episodes. |
| `--seed` | `0` | Random seed. |
| `--print-every` | `10` | Print training statistics every N episodes. |
| `--eval-every` | `100` | Run evaluation every N episodes. |
| `--eval-episodes` | `5` | Number of episodes used for each evaluation. |
| `--render` | off | Render training episodes in a human window. |
| `--out` | auto-generated | Output prefix for the saved learning curve. |
| `--no-show` | off | Save the plot without opening an interactive plot window. |
