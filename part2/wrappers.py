"""Wrappers for environment augmentation used in PPO/SB3 training.

At the moment the module provides a single reward wrapper:

- `RewardShapingWrapper`: adds a small time penalty and a close-to-goal bonus.

The wrapper does not change the physics or the transition dynamics of the
environment. It only changes the scalar reward seen by the agent.
"""

import gymnasium as gym
import numpy as np


class RewardShapingWrapper(gym.RewardWrapper):
    """Shape the reward with a small penalty and a proximity bonus.

    This wrapper expects the underlying task to expose `get_achieved_goal()` and
    a `goal` attribute on `env.unwrapped.task` (as used by panda_gym Push task).

    Logical effect on the reward:

    - start from the original environment reward
    - subtract a tiny time penalty at every step
    - compute the distance between the object and the goal
    - add a bonus when the object is close enough to the goal

    This makes the signal slightly denser than a purely terminal sparse reward,
    because the agent receives a small per-step cost and a proximity-based bonus.
    It is still not a fully dense distance reward, unless you add a graded term
    proportional to the distance itself.
    """

    def __init__(self, env, bonus_distance=0.05, bonus=1.0, time_penalty=1e-3):
        super().__init__(env)
        self.bonus_distance = float(bonus_distance)
        self.bonus = float(bonus)
        self.time_penalty = float(time_penalty)

    def _get_distance(self):
        try:
            task = self.env.unwrapped.task
            achieved = task.get_achieved_goal()
            goal = getattr(task, "goal", None)
            if goal is None:
                return None
            return float(np.linalg.norm(np.asarray(achieved) - np.asarray(goal)))
        except Exception:
            return None

    def reward(self, reward):
        """Return the shaped reward seen by the agent.

        Current formula:

        shaped_reward = original_reward - time_penalty + close_goal_bonus

        where `close_goal_bonus` is applied only when the object-goal distance
        is below `bonus_distance`.
        """
        shaped = float(reward) - self.time_penalty
        dist = self._get_distance()
        if dist is None:
            return shaped

        if dist < self.bonus_distance:
            shaped += self.bonus

        return shaped
