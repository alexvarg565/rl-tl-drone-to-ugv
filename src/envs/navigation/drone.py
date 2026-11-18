"""Obstacle-free planar UAV navigation with world-frame acceleration controls."""

from math import dist, isfinite

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .scenarios import Scenario, ScenarioConfig, generate_scenario, validate_scenario


class DroneNavigationEnv(gym.Env):
    """A point UAV with state [x, y, vx, vy] in meters and seconds.

    Actions are normalized world-X/Y accelerations. Velocity is updated and
    clipped per axis before position is integrated. Observations append the
    goal coordinates to the physical state and are returned as float32.

    Bounds and physical settings are fixed for an environment's lifetime.
    Scenario clearance is an initialization margin, not a runtime footprint.
    Obstacles and rendering are unsupported in this checkpoint.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        x_bounds: tuple[float, float] = (-10.0, 10.0),
        y_bounds: tuple[float, float] = (-10.0, 10.0),
        dt: float = 0.1,
        max_acceleration: float = 1.0,
        max_axis_speed: float = 2.0,
        goal_radius: float = 0.5,
        max_steps: int = 200,
        scenario_config: ScenarioConfig = ScenarioConfig(),
        max_sampling_attempts: int = 1000,
    ):
        super().__init__()
        self.render_mode = None

        for name, value in (("x_bounds", x_bounds), ("y_bounds", y_bounds)):
            bounds = np.asarray(value, dtype=np.float64)
            if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
                raise ValueError(f"{name} must contain exactly two finite values")
            if bounds[0] >= bounds[1]:
                raise ValueError(f"{name} minimum must be less than maximum")
            setattr(self, name, tuple(float(v) for v in bounds))

        for name, value in (
            ("dt", dt), ("max_acceleration", max_acceleration),
            ("max_axis_speed", max_axis_speed), ("goal_radius", goal_radius),
        ):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be finite and positive")
            value = float(value)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)

        for name, value in (
            ("max_steps", max_steps), ("max_sampling_attempts", max_sampling_attempts),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive Python integer")
            setattr(self, name, value)

        if not isinstance(scenario_config, ScenarioConfig):
            raise TypeError("scenario_config must be a ScenarioConfig")
        for name in ("min_start_goal_distance", "clearance"):
            value = getattr(scenario_config, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if any(
            lower + scenario_config.clearance > upper - scenario_config.clearance
            for lower, upper in (self.x_bounds, self.y_bounds)
        ):
            raise ValueError("clearance leaves no valid start/goal positions")
        self.scenario_config = scenario_config

        # A terminal position can overshoot a wall by at most one capped step.
        margin = self.max_axis_speed * self.dt
        if not isfinite(self.max_acceleration * self.dt):
            raise ValueError("max_acceleration * dt must be finite")
        xmin, xmax = self.x_bounds
        ymin, ymax = self.y_bounds
        speed = self.max_axis_speed
        low = np.array([xmin - margin, ymin - margin, -speed, -speed, xmin, ymin])
        high = np.array([xmax + margin, ymax + margin, speed, speed, xmax, ymax])
        float32_max = np.finfo(np.float32).max
        if not (
            np.all(np.isfinite(low)) and np.all(np.isfinite(high))
            and np.all(low >= -float32_max) and np.all(high <= float32_max)
        ):
            raise ValueError("world bounds and speed * dt must allow finite float32 observations")
        # Round outward so float32 terminal observations remain inside the Box.
        with np.errstate(over="ignore"):
            low = np.nextafter(low.astype(np.float32), np.float32(-np.inf))
            high = np.nextafter(high.astype(np.float32), np.float32(np.inf))
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError("observation bounds must remain finite after float32 rounding")
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.scenario = None
        self.state = None
        self.previous_distance = None
        self.step_count = 0
        self._needs_reset = True

    def _get_obs(self):
        return np.array((*self.state, *self.scenario.goal), dtype=np.float32)

    def _get_info(self, distance, *, success=False, out_of_bounds=False):
        return {
            "distance_to_goal": distance,
            "success": success,
            "out_of_bounds": out_of_bounds,
            "step_count": self.step_count,
        }

    def reset(self, *, seed=None, options=None):
        """Generate a scenario, or validate options['scenario'] for fixed replay.

        An unsuccessful reset leaves stepping disabled until a reset succeeds.
        Starts already within goal_radius are evaluated on the first step.
        """
        super().reset(seed=seed)
        self._needs_reset = True
        if options is not None and not isinstance(options, dict):
            raise ValueError("options must be a dictionary or None")
        options = {} if options is None else options
        if set(options) - {"scenario"}:
            raise ValueError("only 'scenario' is supported in reset options")

        if "scenario" in options:
            scenario = options["scenario"]
            if not isinstance(scenario, Scenario):
                raise TypeError("options['scenario'] must be a Scenario")
            if scenario.x_bounds != self.x_bounds or scenario.y_bounds != self.y_bounds:
                raise ValueError("fixed scenario must use the configured world bounds")
            if scenario.obstacles:
                raise ValueError("obstacles are unsupported until runtime collision handling is implemented")
            validate_scenario(scenario, self.scenario_config)
        else:
            scenario = generate_scenario(
                self.np_random, x_bounds=self.x_bounds, y_bounds=self.y_bounds,
                config=self.scenario_config, max_attempts=self.max_sampling_attempts,
            )

        self.scenario = scenario
        # Integrate in float64; cast only the externally returned observations.
        self.state = np.array((*scenario.start, 0.0, 0.0), dtype=np.float64)
        self.step_count = 0
        self.previous_distance = dist(scenario.start, scenario.goal)
        self._needs_reset = False
        return self._get_obs(), self._get_info(self.previous_distance)

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("call reset() before stepping or after episode completion")

        try:
            action = np.asarray(action)
        except (TypeError, ValueError) as error:
            raise ValueError("action must contain two finite real numbers") from error
        if action.shape != (2,) or action.dtype.kind not in "fiu":
            raise ValueError("action must contain two finite real numbers with shape (2,)")
        if not np.all(np.isfinite(action)):
            raise ValueError("action must contain two finite real numbers")
        # Clip before converting so even very large finite values are supported.
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        self.state[2:] = np.clip(
            self.state[2:] + action * (self.max_acceleration * self.dt),
            -self.max_axis_speed, self.max_axis_speed,
        )
        self.state[:2] += self.state[2:] * self.dt
        self.step_count += 1

        distance = dist(self.state[:2], self.scenario.goal)
        x, y = self.state[:2]
        out_of_bounds = bool(
            x < self.x_bounds[0] or x > self.x_bounds[1]
            or y < self.y_bounds[0] or y > self.y_bounds[1]
        )
        success = not out_of_bounds and distance < self.goal_radius
        terminated = out_of_bounds or success
        truncated = self.step_count >= self.max_steps and not terminated

        reward = 10.0 * (self.previous_distance - distance) - 0.01 * float(action @ action)
        if out_of_bounds:
            reward -= 50.0
        elif success:
            reward += 100.0
        self.previous_distance = distance
        self._needs_reset = terminated or truncated
        return (
            self._get_obs(), float(reward), terminated, truncated,
            self._get_info(distance, success=success, out_of_bounds=out_of_bounds),
        )
