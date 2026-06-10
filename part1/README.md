# Part 1: Hopper With From-Scratch Policy Gradients

Part 1 implements policy-gradient algorithms in PyTorch and trains them on the continuous-control MuJoCo `Hopper-v4` environment from Gymnasium.

Implemented algorithms:

- REINFORCE without a baseline.
- REINFORCE with a constant scalar baseline.
- One-step Actor-Critic with a learned state-value critic.

Run all commands from the repository root unless noted otherwise.

## Files

```text
part1/
├── agent.py                         # Policy, critic, return computation, update rules
├── train.py                         # Main Hopper training and evaluation loop
├── test_random_policy.py            # Environment inspection and rendering helper
├── utils/
│   ├── run_part1_experiments.py     # Multi-seed experiment launcher
│   └── plot_comparison.py           # Mean/std comparison plots from saved .npz results
├── results/                         # Saved train/eval metrics
├── plots/                           # Learning curves and comparison figures
├── checkpoints/                     # Existing checkpoint artifacts
└── colab_template/                  # Colab starter notebook
```

## Environment

Install the project dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Conda:

```bash
conda activate <env-name>
python -m pip install -r requirements.txt
```

The training script creates `part1/plots/` and `part1/results/` automatically.

## Task 1: Inspect Hopper

Render a random policy and print the observation/action spaces:

```bash
python part1/test_random_policy.py
```

Use this non-rendering snippet to inspect MuJoCo model details that are useful for the report:

```bash
python - <<'PY'
import gymnasium as gym

env = gym.make("Hopper-v4", render_mode="rgb_array")
env.reset(seed=0)
model = env.unwrapped.model

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

`Hopper-v4` uses continuous observations and continuous actions. The policy therefore outputs a Gaussian distribution over the action vector, and sampled actions are clipped to the environment action bounds before stepping the simulator.

## Training Commands

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

The `--baseline` value is subtracted from discounted returns before computing the policy-gradient loss.

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

### Actor-Critic

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

Compare the three configurations in terms of reward, wall-clock time, training stability, and convergence speed.

## Multi-Seed Experiments

Run the standard experiment set over the default seeds `0 42 99`:

```bash
python part1/utils/run_part1_experiments.py --episodes 5000 --plot
```

Specify a custom seed set:

```bash
python part1/utils/run_part1_experiments.py \
  --seeds 0 1 2 \
  --episodes 3000 \
  --eval-every 100 \
  --eval-episodes 5 \
  --print-every 50 \
  --plot
```

Generate comparison plots from existing results only:

```bash
python part1/utils/plot_comparison.py --seeds 0 42 99
```

The comparison utility expects these result files:

```text
part1/results/reinforce_no_baseline_seed_<seed>.npz
part1/results/reinforce_baseline_20_seed_<seed>.npz
part1/results/actor_critic_seed_<seed>.npz
```

## Outputs

Each `part1/train.py` run saves:

```text
part1/results/<out>.npz
part1/plots/<out>_learning_curve.png
```

The `.npz` file contains:

- `train_returns`: cumulative reward for every training episode.
- `eval_episodes`: episode indices where evaluation was run.
- `eval_returns`: mean deterministic evaluation return.
- `algorithm`, `baseline`, `seed`: run metadata.

Comparison plotting writes:

```text
part1/plots/comparison_train_mean_std.png
part1/plots/comparison_eval_mean_std.png
```

## Implementation Notes

`agent.py` defines a shared `Policy` module with:

- an actor MLP that outputs the mean of a Normal action distribution;
- a learned action standard deviation parameter transformed with `softplus`;
- a critic MLP that outputs one scalar state-value estimate.

For REINFORCE, `discount_rewards` computes full-episode discounted returns with `gamma = 0.99`, and the loss is:

```text
loss = -sum((G_t - baseline) * log pi(a_t | s_t))
```

For Actor-Critic, the update computes a one-step bootstrapped target:

```text
target_t = r_t + gamma * V(s_{t+1}) * (1 - done_t)
advantage_t = target_t - V(s_t)
```

The actor minimizes `-log pi(a_t | s_t) * advantage_t`; the critic minimizes MSE against the bootstrapped target. Advantages are normalized inside each episode before the actor loss.

## `train.py` CLI

| Argument | Default | Description |
| --- | --- | --- |
| `--alg` | `reinforce` | Algorithm: `reinforce` or `actor-critic`. |
| `--baseline` | `0.0` | Constant baseline for REINFORCE. |
| `--episodes` | `3000` | Training episodes. |
| `--seed` | `0` | Base random seed. Episode `i` resets with `seed + i`. |
| `--print-every` | `10` | Print recent training return every N episodes. |
| `--eval-every` | `100` | Run deterministic evaluation every N episodes. |
| `--eval-episodes` | `5` | Evaluation episodes per evaluation point. |
| `--render` | off | Render training episodes in a human window. |
| `--out` | auto | Output prefix for `.npz` and `.png` files. |
| `--no-show` | off | Save plots without opening an interactive window. |
