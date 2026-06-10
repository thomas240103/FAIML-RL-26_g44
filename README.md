# FAIML-RL-26: Course Project

Official assignment at [Google Doc](https://docs.google.com/document/d/1AXgLXux3l69vDAPLL-UYD3luFOw3JbyR-pLCS2yuNZk/edit?usp=sharing)

## Project Overview

This repository contains the codebase for the FAIML (01VSDWS) Reinforcement Learning course project. The project explores learning RL control policies in simulation and studying sim-to-real transfer within a sim-to-sim setting.

The project is divided into two main parts:

- **[Part 1: Hopper Environment](part1/)**: Implementation from scratch of core Reinforcement Learning algorithms (REINFORCE with and without baseline, and Actor-Critic) applied to the continuous control MuJoCo Hopper environment.
- **[Part 2: Panda Push Task (Sim-to-Sim Transfer)](part2/)**: Application of advanced off-the-shelf RL algorithms (PPO, SAC from Stable-Baselines3) to a robotic manipulation task (`panda-gym`). This part focuses on zero-shot transfer from a source environment to a target environment using Domain Randomization strategies (Uniform Domain Randomization - UDR, and Active Domain Randomization - ADR).

## Getting Started

Before starting to implement your own code, make sure to:
1. Read and study the material provided in the assignment.
2. Read the documentation of the main packages you will be using: [Gymnasium](https://gymnasium.farama.org) and [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html).
3. Play around with the code in the template to familiarize yourself with the tools, especially the `test_random_policy.py` scripts.

### 1. Local Environment Setup

If you have a Linux or macOS system, you can work on the course project directly on your local machine. This allows you to render the environments and visualize the trained policies in action. We highly suggest using `venv` or Conda to manage the Python environment.

**Dependencies:**
```bash
# Using venv
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Extra step for Part 2 (Panda-Gym Push task):**
To train on the panda-gym task, you must install the custom local version of `panda-gym`:
```bash
cd part2/panda-gym
pip install -e .
```

Check your installation by launching:
```bash
python part1/test_random_policy.py
```

### 2. Google Colab (Alternative)

You can also run the code on [Google Colab](https://colab.research.google.com/).
- Download all files contained in the `part1/colab_template` folder.
- Load the `test_random_policy.ipynb` file on Colab and follow the instructions.

*NOTE 1:* Rendering is currently **not** officially supported on Colab. We recommend running the visual interface locally at least once to understand the underlying environments.
*NOTE 2:* You need to stay connected to the Google Colab interface at all times for your python scripts to keep training.

## Documentation & Useful Links

For exhaustive documentation and task-specific commands, refer to the following files:

- 📖 **[Part 1 Instructions & Commands](part1/README.md)**: Details on training REINFORCE and Actor-Critic models on the Hopper environment.
- 📖 **[Part 2 Instructions & Commands](part2/README.md)**: Details on training PPO/SAC models on the Panda Push task and using Domain Randomization.
- 📝 **[Project TODOs & Experiment Plan](todo.md)**: Comprehensive checklist of the required experiments, recommended step budgets, and evaluation settings for the report.
- 📄 **[Report Structure Guidelines](report_info.md)**: Detailed breakdown of the expected 5-page report, including the mandatory tables and analytical sections.

## Project Structure

```text
FAIML-RL-26/
├── README.md               <-- Main entry point (this file)
├── requirements.txt        <-- Python dependencies
├── todo.md                 <-- Comprehensive experiment plan and tasks
├── report_info.md          <-- Guidelines for the final report
├── models/                 <-- Saved trained models
├── ppo_logs/ & sac_logs/ & tb_logs/ <-- TensorBoard logs
│
├── part1/                  <-- Part 1: Hopper (Custom Algorithms)
│   ├── README.md           <-- Documentation for Part 1
│   ├── agent.py            <-- Algorithm implementations (REINFORCE, Actor-Critic)
│   ├── train.py            <-- Training script for Part 1
│   ├── test_random_policy.py
│   ├── run_part1_experiments.py
│   ├── plot_comparison.py
│   └── colab_template/
│
└── part2/                  <-- Part 2: Panda Push Task (Sim-to-Sim)
    ├── README.md           <-- Documentation and commands for Part 2
    ├── docs/               <-- Comprehensive documentation for each Part 2 script
    │   └── README.md
    ├── train_sb3.py        <-- Main training script for SAC/PPO
    ├── train_sb3_ppo.py    <-- PPO specific training script
    ├── train_sb3_sac.py    <-- SAC specific training script
    ├── eval_sb3.py         <-- Evaluation script for transfer learning
    ├── rand_wrapper.py     <-- Domain randomization wrappers (UDR/ADR)
    ├── wrappers.py         <-- Other environment wrappers
    ├── test_random_policy.py
    └── panda-gym/          <-- Customized panda-gym environment
        └── panda_gym/
            └── envs/
                └── tasks/
                    └── push.py <-- The target environment
```