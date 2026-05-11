# Part 1 CLI Help

This file summarizes how to launch the Hopper experiments from the command line.

## Environment

Use the conda environment with the project dependencies installed:

```bash
conda activate venv
```

Alternatively, prefix every command with:

```bash
conda run -n venv
```

## Basic Usage

Run `train.py` from the repository root:

```bash
python part1/train.py --alg reinforce --no-show
```

The `--no-show` flag saves the learning curve without opening an interactive plot window. This is recommended for long experiments or terminal-only runs.

## Main Experiments

### REINFORCE without baseline

```bash
python part1/train.py \
  --alg reinforce \
  --episodes 3000 \
  --seed 0 \
  --eval-every 100 \
  --eval-episodes 5 \
  --out reinforce_no_baseline_seed_0 \
  --no-show
```

### REINFORCE with constant baseline

```bash
python part1/train.py \
  --alg reinforce \
  --baseline 100 \
  --episodes 3000 \
  --seed 0 \
  --eval-every 100 \
  --eval-episodes 5 \
  --out reinforce_baseline_100_seed_0 \
  --no-show
```

### Actor-Critic

```bash
python part1/train.py \
  --alg actor-critic \
  --episodes 3000 \
  --seed 0 \
  --eval-every 100 \
  --eval-episodes 5 \
  --out actor_critic_seed_0 \
  --no-show
```

## One-Line Commands With Conda

If you do not want to activate the environment first:

```bash
conda run -n venv python part1/train.py --alg reinforce --episodes 3000 --seed 0 --out reinforce_no_baseline_seed_0 --no-show
```

```bash
conda run -n venv python part1/train.py --alg reinforce --baseline 20 --episodes 3000 --seed 0 --out reinforce_baseline_20_seed_0 --no-show
```

```bash
conda run -n venv python part1/train.py --alg actor-critic --episodes 3000 --seed 0 --out actor_critic_seed_0 --no-show
```

## CLI Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--alg` | `reinforce` | Training algorithm. Choices: `reinforce`, `actor-critic`. |
| `--baseline` | `0.0` | Constant baseline subtracted from discounted returns. Used by REINFORCE. |
| `--episodes` | `3000` | Number of training episodes. |
| `--seed` | `0` | Random seed for reproducibility. |
| `--print-every` | `10` | Print training statistics every N episodes. |
| `--eval-every` | `100` | Run evaluation every N episodes. |
| `--eval-episodes` | `5` | Number of episodes used for each evaluation. |
| `--render` | off | Render the training environment in a human window. |
| `--out` | auto-generated | Prefix for the saved learning curve image. |
| `--no-show` | off | Save the plot without opening an interactive window. |

## Output

Each run saves a learning curve under `part1/`:

```text
part1/<out>_learning_curve.png
```

For example:

```text
part1/reinforce_no_baseline_seed_0_learning_curve.png
part1/reinforce_baseline_100_seed_0_learning_curve.png
part1/actor_critic_seed_0_learning_curve.png
```

## Suggested Report Workflow

Run at least the three main experiments:

1. REINFORCE without baseline.
2. REINFORCE with a constant baseline.
3. Actor-Critic.

Then compare:

- final evaluation return,
- learning stability,
- convergence speed,
- training time,
- effect of the baseline on REINFORCE variance.

For a more reliable comparison, repeat each experiment with multiple seeds, for example `0`, `1`, and `2`.
