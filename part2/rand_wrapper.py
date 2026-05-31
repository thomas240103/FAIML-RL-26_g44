from collections import deque

import gymnasium as gym
import numpy as np


class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that randomizes physical parameters at reset time.

    Modes:
    - `none`: keep the nominal environment mass unchanged
    - `udr`: sample uniformly from the full mass/friction ranges
    - `adr`: sample from an adaptive range that expands/shrinks based on success
    """

    def __init__(
        self,
        env,
        mass_range=(1.0, 5.0),
        object_lateral_friction_range=(0.5, 1.2),
        table_lateral_friction_range=(0.5, 1.2),
        object_spinning_friction_range=(0.0, 0.005),
        mode="none",
        adr_window=20,
        adr_high_threshold=0.70,
        adr_low_threshold=0.35,
        adr_step=0.25,
    ):
        super().__init__(env)

        if mode not in {"none", "udr", "adr"}:
            raise ValueError(f"Unsupported randomization mode: {mode}")

        mass_min_limit, mass_max_limit = mass_range
        if mass_min_limit <= 0 or mass_min_limit > mass_max_limit:
            raise ValueError(f"Invalid mass_range: {mass_range}")
        self._validate_nonnegative_range(
            "object_lateral_friction_range",
            object_lateral_friction_range,
        )
        self._validate_nonnegative_range(
            "table_lateral_friction_range",
            table_lateral_friction_range,
        )
        self._validate_nonnegative_range(
            "object_spinning_friction_range",
            object_spinning_friction_range,
        )

        self.mode = mode
        self.mass_range = mass_range
        self.object_lateral_friction_range = object_lateral_friction_range
        self.table_lateral_friction_range = table_lateral_friction_range
        self.object_spinning_friction_range = object_spinning_friction_range

        # Global limits allowed by domain randomization.
        self.mass_min_limit = float(mass_min_limit)
        self.mass_max_limit = float(mass_max_limit)
        self.object_lateral_friction_min, self.object_lateral_friction_max = map(
            float,
            object_lateral_friction_range,
        )
        self.table_lateral_friction_min, self.table_lateral_friction_max = map(
            float,
            table_lateral_friction_range,
        )
        self.object_spinning_friction_min, self.object_spinning_friction_max = map(
            float,
            object_spinning_friction_range,
        )

        # Nominal mass comes from the underlying environment definition.
        self.nominal_mass = float(self.env.unwrapped.task.current_mass)
        nominal_friction = self._get_nominal_friction()
        self.nominal_object_lateral_friction = nominal_friction["object_lateral"]
        self.nominal_table_lateral_friction = nominal_friction["table_lateral"]
        self.nominal_object_spinning_friction = nominal_friction["object_spinning"]

        # Current ADR sampling ranges (start at nominal, widen on success).
        self.mass_min = self.nominal_mass
        self.mass_max = self.nominal_mass
        self.obj_lat_fric_min = self.nominal_object_lateral_friction
        self.obj_lat_fric_max = self.nominal_object_lateral_friction
        self.table_lat_fric_min = self.nominal_table_lateral_friction
        self.table_lat_fric_max = self.nominal_table_lateral_friction
        self.obj_spin_fric_min = self.nominal_object_spinning_friction
        self.obj_spin_fric_max = self.nominal_object_spinning_friction

        # Keep a short success history to adapt the ADR range.
        self.success_history = deque(maxlen=adr_window)
        self.adr_window = adr_window
        self.adr_high_threshold = adr_high_threshold
        self.adr_low_threshold = adr_low_threshold
        self.adr_step = float(adr_step)

        self.last_sample_type = "fixed"
        self.last_mass = self.nominal_mass
        self.last_object_lateral_friction = self.nominal_object_lateral_friction
        self.last_table_lateral_friction = self.nominal_table_lateral_friction
        self.last_object_spinning_friction = self.nominal_object_spinning_friction

        if self.mode == "udr":
            self.mass_min = self.mass_min_limit
            self.mass_max = self.mass_max_limit
            self.obj_lat_fric_min = self.object_lateral_friction_min
            self.obj_lat_fric_max = self.object_lateral_friction_max
            self.table_lat_fric_min = self.table_lateral_friction_min
            self.table_lat_fric_max = self.table_lateral_friction_max
            self.obj_spin_fric_min = self.object_spinning_friction_min
            self.obj_spin_fric_max = self.object_spinning_friction_max
        elif self.mode == "adr":
            self.mass_min = self.nominal_mass
            self.mass_max = self.nominal_mass

    @staticmethod
    def _validate_nonnegative_range(name: str, value_range) -> None:
        low, high = value_range
        if low < 0 or high < 0 or low > high:
            raise ValueError(f"Invalid {name}: {value_range}")

    def _get_dynamics_info(self, body_name: str) -> tuple:
        sim = self.env.unwrapped.task.sim
        body_id = sim._bodies_idx[body_name]
        return sim.physics_client.getDynamicsInfo(body_id, -1)

    def _get_nominal_friction(self) -> dict[str, float]:
        object_info = self._get_dynamics_info("object")
        table_info = self._get_dynamics_info("table")
        return {
            "object_lateral": float(object_info[1]),
            "table_lateral": float(table_info[1]),
            "object_spinning": float(object_info[7]),
        }

    def _uniform(self, low: float, high: float) -> float:
        rng = getattr(self.env.unwrapped, "np_random", None)
        if rng is None:
            return float(np.random.uniform(low, high))
        return float(rng.uniform(low, high))

    def _sample_mass(self):
        if self.mode == "none":
            self.last_sample_type = "fixed"
            self.last_mass = self.nominal_mass
            return None

        if self.mode == "udr":
            self.last_sample_type = "uniform"
            self.last_mass = self._uniform(self.mass_min_limit, self.mass_max_limit)
            return self.last_mass

        if self.mode == "adr":
            self.last_sample_type = "adaptive"
            self.last_mass = self._uniform(self.mass_min, self.mass_max)
            return self.last_mass

        raise NotImplementedError(f"Sampling strategy '{self.mode}' is not implemented.")

    def _sample_friction(self):
        if self.mode == "none":
            self.last_object_lateral_friction = self.nominal_object_lateral_friction
            self.last_table_lateral_friction = self.nominal_table_lateral_friction
            self.last_object_spinning_friction = self.nominal_object_spinning_friction
            return None

        # udr: ranges are set to full limits at construction; adr: ranges adapt over time.
        self.last_object_lateral_friction = self._uniform(self.obj_lat_fric_min, self.obj_lat_fric_max)
        self.last_table_lateral_friction = self._uniform(self.table_lat_fric_min, self.table_lat_fric_max)
        self.last_object_spinning_friction = self._uniform(self.obj_spin_fric_min, self.obj_spin_fric_max)
        return {
            "object_lateral": self.last_object_lateral_friction,
            "table_lateral": self.last_table_lateral_friction,
            "object_spinning": self.last_object_spinning_friction,
        }

    def _update_adr_range(self) -> None:
        if self.mode != "adr" or len(self.success_history) < self.adr_window:
            return

        success_rate = float(np.mean(self.success_history))
        self.success_history.clear()

        mass_step = self.adr_step * (self.mass_max_limit - self.nominal_mass)
        obj_lat_step = self.adr_step * (self.object_lateral_friction_max - self.object_lateral_friction_min)
        table_lat_step = self.adr_step * (self.table_lateral_friction_max - self.table_lateral_friction_min)
        obj_spin_step = self.adr_step * (self.object_spinning_friction_max - self.object_spinning_friction_min)

        if success_rate >= self.adr_high_threshold:
            # mass: one-sided — only expand upward toward target domain
            self.mass_max = min(self.mass_max_limit, self.mass_max + mass_step)
            # friction: two-sided
            self.obj_lat_fric_min = max(self.object_lateral_friction_min, self.obj_lat_fric_min - obj_lat_step)
            self.obj_lat_fric_max = min(self.object_lateral_friction_max, self.obj_lat_fric_max + obj_lat_step)
            self.table_lat_fric_min = max(self.table_lateral_friction_min, self.table_lat_fric_min - table_lat_step)
            self.table_lat_fric_max = min(self.table_lateral_friction_max, self.table_lat_fric_max + table_lat_step)
            self.obj_spin_fric_min = max(self.object_spinning_friction_min, self.obj_spin_fric_min - obj_spin_step)
            self.obj_spin_fric_max = min(self.object_spinning_friction_max, self.obj_spin_fric_max + obj_spin_step)
        elif success_rate <= self.adr_low_threshold:
            self.mass_max = max(self.nominal_mass, self.mass_max - mass_step)
            self.obj_lat_fric_min = min(self.nominal_object_lateral_friction, self.obj_lat_fric_min + obj_lat_step)
            self.obj_lat_fric_max = max(self.nominal_object_lateral_friction, self.obj_lat_fric_max - obj_lat_step)
            if self.obj_lat_fric_min > self.obj_lat_fric_max:
                self.obj_lat_fric_min = self.obj_lat_fric_max = self.nominal_object_lateral_friction
            self.table_lat_fric_min = min(self.nominal_table_lateral_friction, self.table_lat_fric_min + table_lat_step)
            self.table_lat_fric_max = max(self.nominal_table_lateral_friction, self.table_lat_fric_max - table_lat_step)
            if self.table_lat_fric_min > self.table_lat_fric_max:
                self.table_lat_fric_min = self.table_lat_fric_max = self.nominal_table_lateral_friction
            self.obj_spin_fric_min = min(self.nominal_object_spinning_friction, self.obj_spin_fric_min + obj_spin_step)
            self.obj_spin_fric_max = max(self.nominal_object_spinning_friction, self.obj_spin_fric_max - obj_spin_step)
            if self.obj_spin_fric_min > self.obj_spin_fric_max:
                self.obj_spin_fric_min = self.obj_spin_fric_max = self.nominal_object_spinning_friction

        print(
            f"[adr:update] success_rate={success_rate:.2f} "
            f"mass=[{self.mass_min:.2f},{self.mass_max:.2f}] "
            f"obj_lat=[{self.obj_lat_fric_min:.4f},{self.obj_lat_fric_max:.4f}] "
            f"table_lat=[{self.table_lat_fric_min:.4f},{self.table_lat_fric_max:.4f}] "
            f"obj_spin=[{self.obj_spin_fric_min:.6f},{self.obj_spin_fric_max:.6f}]"
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.mode == "adr" and (terminated or truncated):
            success = 0.0
            if isinstance(info, dict) and "is_success" in info:
                success = float(info["is_success"])
            self.success_history.append(success)
            self._update_adr_range()

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        new_mass = self._sample_mass()
        new_friction = self._sample_friction()

        if new_mass is not None:
            self._set_object_mass(new_mass)

        if new_friction is not None:
            self._set_friction(new_friction)

        if new_mass is not None:
            friction_text = ""
            if new_friction is not None:
                friction_text = (
                    f" object_mu={new_friction['object_lateral']:.4f}"
                    f" table_mu={new_friction['table_lateral']:.4f}"
                    f" object_spin={new_friction['object_spinning']:.4f}"
                )

            print(
                f"[{self.mode}] mass={new_mass:.2f} "
                f"range=[{self.mass_min:.2f},{self.mass_max:.2f}] "
                f"type={self.last_sample_type}"
                f"{friction_text}"
            )

        return super().reset(**kwargs)

    def _set_object_mass(self, mass: float) -> None:
        task = self.env.unwrapped.task
        sim = task.sim
        object_body_id = sim._bodies_idx["object"]

        sim.physics_client.changeDynamics(
            bodyUniqueId=object_body_id,
            linkIndex=-1,
            mass=float(mass),
        )
        task.current_mass = float(mass)

    def _set_friction(self, friction: dict[str, float]) -> None:
        sim = self.env.unwrapped.task.sim
        sim.set_lateral_friction("object", -1, friction["object_lateral"])
        sim.set_lateral_friction("table", -1, friction["table_lateral"])
        sim.set_spinning_friction("object", -1, friction["object_spinning"])
