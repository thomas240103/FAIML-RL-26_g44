# Part 2: Panda Push Task (Sim-to-Sim Transfer)

This part focuses on utilizing off-the-shelf algorithms from Stable-Baselines3 (SAC and PPO) to train a robotic arm (`panda-gym`) on a Push task. We aim to perform Sim-to-Sim transfer, testing Domain Randomization strategies to bridge the reality gap.

## Code Overview & Documentation

For an exhaustive explanation of every script, we have a dedicated `docs/` folder. Please refer to **[docs/README.md](docs/README.md)** as the main entry point.

Here is a quick overview of the main components:

- **[`train_sb3.py`](docs/train_sb3.md)**, **[`train_sb3_ppo.py`](docs/train_sb3_ppo.md)**, **[`train_sb3_sac.py`](docs/train_sb3_sac.md)**: Scripts to train the agent using SAC or PPO. They parse arguments such as `--sampling-strategy` (none, udr, adr) and `--env-type` (source or target), and set up the training environment accordingly.
- **[`run_tests.py`](docs/run_tests.md)**: Script for executing automated grid searches and multiple profiles.
- **[`eval_sb3.py`](docs/eval_sb3.md)**: A comprehensive evaluation script that tests trained models (`.zip`) on the target environment. It computes summary statistics (mean return, success rate) over customized sets of masses and friction coefficients.
- **[`rand_wrapper.py`](docs/rand_wrapper.md)**: The core component for Domain Randomization. It implements `RandomizationWrapper`, a Gym wrapper that alters the environment's mass and friction parameters at reset time, supporting `none` (no randomization), `udr` (Uniform Domain Randomization), and `adr` (Active Domain Randomization).
- **[`wrappers.py`](docs/wrappers.md)**: Additional environment wrappers.
- **[`test_random_policy.py`](docs/test_random_policy.md)**: Script for testing the environment with a random policy.
- **[`research_directions.md`](docs/research_directions.md)**: Contains literature survey and research directions.

---
