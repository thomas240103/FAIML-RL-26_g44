# Part 2: Panda Push Sim-to-Sim Transfer

Part 2 trains Stable-Baselines3 agents on a customized `panda-gym` `PandaPush-v3` task. The main objective is source-to-target transfer: train on a source simulation and evaluate on a target simulation with different object dynamics.

The customized task defines:

- `source`: object mass is `1.0 kg`;
- `target`: object mass is `5.0 kg`;
- dense reward by default in the project scripts;
- 3D end-effector control through the Panda robot;
- randomized object start position and a fixed goal at the table center.

Run all commands from the repository root unless noted otherwise.

## Files

```text
part2/
├── train_sb3.py              # Main PPO/SAC training entry point
├── train_sb3_ppo.py          # Older PPO-only trainer
├── train_sb3_sac.py          # Older SAC-only trainer
├── eval_sb3.py               # Evaluation on source/target with mass/friction schedules
├── rand_wrapper.py           # none, UDR, and ADR physics randomization
├── wrappers.py               # RewardShapingWrapper
├── run_tests.py              # PPO/SAC grid-search runner
├── test_random_policy.py     # Environment inspection/rendering helper
├── plots/
├── ppo_logs/
└── panda-gym/                # Local editable panda-gym package
```

`train_sb3.py` is the recommended training script because it supports both algorithms and all current randomization options. The PPO-only and SAC-only scripts are kept for compatibility with earlier experiments.

## Environment

Install dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the local `panda-gym` package was not installed through `requirements.txt`, install it manually:

```bash
python -m pip install -e ./part2/panda-gym
```

Check the environment registration and render a random policy:

```bash
python part2/test_random_policy.py
```

## Domain Randomization

`rand_wrapper.py` applies randomization at environment reset time.

| Mode | Behavior |
| --- | --- |
| `none` | Keep the nominal source or target physics. |
| `udr` | Uniformly sample mass and friction from the configured ranges every episode. |
| `adr` | Start from nominal physics and adapt the sampling range using recent episode success rate. |

Current randomized parameters:

- object mass;
- object lateral friction;
- table lateral friction;
- object spinning friction.

Default training ranges in `train_sb3.py` are:

| Parameter | Default Range |
| --- | --- |
| object mass | `[1.0, 5.0] kg` |
| object lateral friction | `[0.5, 1.2]` |
| table lateral friction | `[0.5, 1.2]` |
| object spinning friction | `[0.0, 0.005]` |

For UDR, the wrapper samples uniformly from the full ranges. For ADR, the wrapper keeps a sliding success window, expands the range when success is high, and contracts it when success is low.

## Training With `train_sb3.py`

### SAC Baseline On Source

```bash
python part2/train_sb3.py \
  --algorithm sac \
  --env-type source \
  --sampling-strategy none \
  --timesteps 500000 \
  --seed 42 \
  --save \
  --run-name SAC_none_source_500k
```

Saved model:

```text
models/SAC_none_source_500k/model.zip
```

TensorBoard log:

```text
tb_logs/SAC_none_source_500k_1/
```

### SAC With UDR

```bash
python part2/train_sb3.py \
  --algorithm sac \
  --env-type source \
  --sampling-strategy udr \
  --timesteps 500000 \
  --seed 42 \
  --mass-min 1.0 \
  --mass-max 5.0 \
  --object-lateral-friction-min 0.5 \
  --object-lateral-friction-max 1.2 \
  --table-lateral-friction-min 0.5 \
  --table-lateral-friction-max 1.2 \
  --object-spinning-friction-min 0.0 \
  --object-spinning-friction-max 0.005 \
  --save \
  --run-name SAC_udr_source_500k
```

### SAC With ADR

```bash
python part2/train_sb3.py \
  --algorithm sac \
  --env-type source \
  --sampling-strategy adr \
  --timesteps 500000 \
  --seed 42 \
  --save \
  --run-name SAC_adr_source_500k
```

### PPO Baseline

```bash
python part2/train_sb3.py \
  --algorithm ppo \
  --env-type source \
  --sampling-strategy none \
  --timesteps 500000 \
  --seed 42 \
  --n-envs 4 \
  --n-steps 1024 \
  --n-epochs 10 \
  --learning-rate 0.0003 \
  --gamma 0.99 \
  --ent-coef 0.001 \
  --save \
  --run-name PPO_none_source_500k
```

## Training CLI Reference

Common arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--algorithm` | required | `ppo` or `sac`. |
| `--sampling-strategy` | `none` | `none`, `udr`, or `adr`. |
| `--env-type` | `source` | `source` or `target`. |
| `--timesteps` | `500000` | SB3 training timesteps. |
| `--seed` | `42` | Environment and model seed. |
| `--save` | off | Save the trained model under `models/<run-name>/model.zip`. |
| `--run-name` | auto | Model directory and TensorBoard run name. |
| `--learning-rate` | algorithm default | Shared PPO/SAC learning rate. |
| `--gamma` | algorithm default | Discount factor. |

Randomization range arguments:

| Argument | Default |
| --- | --- |
| `--mass-min` | `1.0` |
| `--mass-max` | `5.0` |
| `--object-lateral-friction-min` | `0.5` |
| `--object-lateral-friction-max` | `1.2` |
| `--table-lateral-friction-min` | `0.5` |
| `--table-lateral-friction-max` | `1.2` |
| `--object-spinning-friction-min` | `0.0` |
| `--object-spinning-friction-max` | `0.005` |

PPO-specific arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--n-envs` | `1` | Parallel environments. Values greater than 1 use `SubprocVecEnv`. |
| `--ent-coef` | `0.001` | Entropy coefficient. |
| `--n-steps` | `2048` | Rollout steps per environment before an update. |
| `--clip-range` | `0.2` | PPO policy-ratio clipping range. |
| `--gae-lambda` | `0.95` | GAE lambda. |
| `--n-epochs` | `10` | Epochs per rollout batch. |

SAC-specific arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--her` | off | Use `HerReplayBuffer`. |
| `--batch-size` | `256` | Minibatch size. |
| `--gradient-steps` | `1` | Gradient updates per training cycle. |
| `--learning-starts` | `100` | Steps before learning starts. |
| `--buffer-size` | `1000000` | Replay buffer size. |
| `--tau` | `0.005` | Soft target-network update coefficient. |
| `--train-freq` | `1` | Environment steps between training cycles. |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps`. |

## Evaluation

`eval_sb3.py` loads a saved SB3 model and runs episodes on `source` or `target`. It reports per-episode return, success, sampled mass/friction, aggregate return statistics, success rate, and binned summaries by mass/friction.

### Fixed Target Evaluation

```bash
python part2/eval_sb3.py \
  --model-path models/SAC_none_source_500k/model.zip \
  --algo sac \
  --env-type target \
  --episodes 50 \
  --eval-mass-mode fixed \
  --eval-friction-mode fixed \
  --seed 0
```

If `--algo` is omitted, the script tries to infer the algorithm from a filename that starts with `sac_` or `ppo_`. Use `--algo` for model paths such as `models/SAC_none_source_500k/model.zip`, where the filename itself is only `model.zip`.

### Uniform Mass And Friction Stress Test

```bash
python part2/eval_sb3.py \
  --model-path models/SAC_udr_source_500k/model.zip \
  --algo sac \
  --env-type target \
  --episodes 100 \
  --eval-mass-mode uniform \
  --mass-min 5.0 \
  --mass-max 10.0 \
  --eval-friction-mode uniform \
  --object-lateral-friction-min 0.2 \
  --object-lateral-friction-max 1.5 \
  --table-lateral-friction-min 0.2 \
  --table-lateral-friction-max 1.5 \
  --object-spinning-friction-min 0.0 \
  --object-spinning-friction-max 0.01 \
  --seed 0
```

### Grid Evaluation Over Masses

```bash
python part2/eval_sb3.py \
  --model-path models/SAC_none_source_500k/model.zip \
  --algo sac \
  --env-type target \
  --episodes 60 \
  --eval-mass-mode grid \
  --mass-values 5,6,7,8,9,10 \
  --seed 0
```

### Render A Trained Policy

```bash
python part2/eval_sb3.py \
  --model-path models/SAC_none_source_500k/model.zip \
  --algo sac \
  --env-type target \
  --episodes 3 \
  --render \
  --render-delay 0.02
```

## Evaluation CLI Reference

| Argument | Default | Description |
| --- | --- | --- |
| `--model-path` | root-level `sac_push_none_source_500k.zip` | SB3 `.zip` model path. |
| `--algo` | inferred | `sac` or `ppo`. |
| `--episodes` | `50` | Number of evaluation episodes. |
| `--stochastic` | off | Use stochastic predictions instead of deterministic actions. |
| `--render` | off | Render in a human window. |
| `--render-delay` | `0.0` | Sleep between rendered steps. |
| `--env-type` | `target` | Evaluation environment type. |
| `--max-episode-steps` | `500` | Episode horizon. |
| `--seed` | `None` | Optional base seed; episode `i` uses `seed + i`. |
| `--eval-mass-mode` | `fixed` | `fixed`, `uniform`, or `grid`. |
| `--fixed-mass` | nominal | Fixed mass for `fixed` mode. |
| `--mass-min` / `--mass-max` | `5.0` / `10.0` | Range for uniform/grid mass evaluation. |
| `--mass-values` | unset | Comma-separated mass list for grid mode. |
| `--mass-grid-size` | `6` | Grid size when `--mass-values` is unset. |
| `--heavy-mass-threshold` | `8.0` | Threshold for heavy-object success reporting. |
| `--eval-friction-mode` | `fixed` | `fixed` or `uniform`. |
| `--object-lateral-friction` | nominal | Fixed object lateral friction. |
| `--table-lateral-friction` | nominal | Fixed table lateral friction. |
| `--object-spinning-friction` | nominal | Fixed object spinning friction. |
| `--friction-bins` | `5` | Number of bins in friction summaries. |

## Grid Searches With `run_tests.py`

`run_tests.py` launches profile-based parameter sweeps by calling `part2/train_sb3.py`. It writes each run log under:

```text
part2/test/profile_<profile>/<run-name>/training_log.txt
```

Run a quick smoke test:

```bash
cd part2
python run_tests.py --algorithm sac --profile 1 --smoke
python run_tests.py --algorithm ppo --profile 1 --smoke
```

Run a full profile and save models:

```bash
cd part2
python run_tests.py --algorithm sac --profile 5 --save
```

Profiles `1` through `5` are defined directly in `run_tests.py` for SAC and PPO. Smoke tests use short timestep budgets to verify that configurations run without crashing; full profiles use `10_000_000` timesteps per run.

## TensorBoard

Unified `train_sb3.py` writes to `tb_logs/` from the repository root:

```bash
tensorboard --logdir tb_logs
```

The older PPO-only script writes to `part2/ppo_logs/` when run from the repository root:

```bash
tensorboard --logdir part2/ppo_logs
```

## Notes On Older Scripts

`train_sb3_ppo.py` and `train_sb3_sac.py` are narrower historical entry points. Prefer `train_sb3.py` for new experiments because it:

- shares one CLI for PPO and SAC;
- supports configurable mass and friction ranges;
- saves models consistently under `models/<run-name>/model.zip`;
- writes logs consistently under `tb_logs/`.

The older scripts can still be useful for reproducing earlier runs if their exact hyperparameters are needed.
