# FAIML-RL-26: Reinforcement Learning Course Project

This repository contains the code for the FAIML (01VSDWS) Reinforcement Learning course project. It studies continuous-control policy learning in simulation and sim-to-sim transfer for robotic manipulation.

The project has two independent parts:

- [Part 1: Hopper](part1/README.md) implements REINFORCE, REINFORCE with a constant baseline, and one-step Actor-Critic from scratch in PyTorch on `Hopper-v4`.
- [Part 2: Panda Push](part2/README.md) trains Stable-Baselines3 PPO/SAC policies on a customized `panda-gym` `PandaPush-v3` task and evaluates source-to-target transfer with domain randomization.

The original assignment is linked in the course material and is also available as [delivery_instruction.pdf](delivery_instruction.pdf).

## Documentation

- [Part 1 README: Hopper from-scratch algorithms](part1/README.md)
- [Part 2 README: Panda Push sim-to-sim transfer](part2/README.md)

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── delivery_instruction.pdf
├── report/
│   ├── main.tex
│   ├── main.pdf
│   └── sec/
├── models/
│   └── <saved Stable-Baselines3 models>
├── tb_logs/
│   └── <TensorBoard logs from part2/train_sb3.py>
├── part1/
│   ├── README.md
│   ├── agent.py
│   ├── train.py
│   ├── test_random_policy.py
│   ├── utils/
│   │   ├── run_part1_experiments.py
│   │   └── plot_comparison.py
│   ├── results/
│   ├── plots/
│   ├── checkpoints/
│   └── colab_template/
└── part2/
    ├── README.md
    ├── train_sb3.py
    ├── train_sb3_ppo.py
    ├── train_sb3_sac.py
    ├── eval_sb3.py
    ├── rand_wrapper.py
    ├── wrappers.py
    ├── run_tests.py
    ├── test_random_policy.py
    ├── plots/
    ├── ppo_logs/
    └── panda-gym/
```

## Setup

Use Python 3.10+ and install dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the local `part2/panda-gym` package in editable mode. If you install dependencies manually, make sure this package is also installed:

```bash
python -m pip install -e ./part2/panda-gym
```

Smoke-test the two environments:

```bash
python part1/test_random_policy.py
python part2/test_random_policy.py
```

The random-policy scripts render a simulator window. On a headless machine, use the training/evaluation commands in each part with non-rendering settings.

## Part 1 Quick Start

Train the three main Hopper configurations:

```bash
python part1/train.py --alg reinforce --episodes 3000 --seed 0 --out reinforce_no_baseline_seed_0 --no-show
python part1/train.py --alg reinforce --baseline 20 --episodes 3000 --seed 0 --out reinforce_baseline_20_seed_0 --no-show
python part1/train.py --alg actor-critic --episodes 3000 --seed 0 --out actor_critic_seed_0 --no-show
```

Run the multi-seed experiment helper and generate comparison plots:

```bash
python part1/utils/run_part1_experiments.py --seeds 0 42 99 --episodes 5000 --plot
```

Outputs are written to `part1/results/` (`.npz` metrics) and `part1/plots/` (`.png` learning curves and comparisons). See [part1/README.md](part1/README.md) for the full command reference and implementation notes.

## Part 2 Quick Start

Train SAC on the source Panda Push task and save the model:

```bash
python part2/train_sb3.py \
  --algorithm sac \
  --env-type source \
  --sampling-strategy none \
  --timesteps 500000 \
  --save \
  --run-name SAC_none_source_500k
```

Train with uniform domain randomization over mass and friction:

```bash
python part2/train_sb3.py \
  --algorithm sac \
  --env-type source \
  --sampling-strategy udr \
  --timesteps 500000 \
  --mass-min 1.0 \
  --mass-max 5.0 \
  --object-lateral-friction-min 0.5 \
  --object-lateral-friction-max 1.2 \
  --table-lateral-friction-min 0.5 \
  --table-lateral-friction-max 1.2 \
  --save \
  --run-name SAC_udr_source_500k
```

Evaluate a saved model on the target task:

```bash
python part2/eval_sb3.py \
  --model-path models/SAC_none_source_500k/model.zip \
  --algo sac \
  --env-type target \
  --episodes 50 \
  --eval-mass-mode fixed \
  --seed 0
```

See [part2/README.md](part2/README.md) for PPO/SAC options, domain-randomization modes, evaluation schedules, and grid-search utilities.

## Reports And Artifacts

- The LaTeX report source is in `report/`.
- Pretrained Part 2 models are stored in `models/`.
- TensorBoard logs are stored mainly in `tb_logs/`.

Start TensorBoard from the repository root with:

```bash
tensorboard --logdir tb_logs
```
