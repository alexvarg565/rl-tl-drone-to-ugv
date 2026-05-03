import gymnasium as gym
from gymnasium import spaces
import numpy as np


class UGVEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.world_size = 10.0
        self.max_steps = 200
        self.step_count = 0

        # x, y, vx, vy, goal_x, goal_y
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )

        # speed control and steering-like control
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        self.state = None
        self.goal = None
        self.previous_distance = None

    def _get_obs(self):
        return np.array([
            self.state[0],
            self.state[1],
            self.state[2],
            self.state[3],
            self.goal[0],
            self.goal[1]
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.goal = np.array([8.0, 8.0], dtype=np.float32)

        self.previous_distance = np.linalg.norm(self.state[:2] - self.goal)

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0)
        throttle, steer = action

        x, y, vx, vy = self.state

        vx += throttle * 0.08
        vy += steer * 0.05

        # Light velocity clipping helps stop runaway movement.
        vx = np.clip(vx, -1.5, 1.5)
        vy = np.clip(vy, -1.5, 1.5)

        x += vx * 0.1
        y += vy * 0.1

        self.state = np.array([x, y, vx, vy], dtype=np.float32)

        current_distance = np.linalg.norm(self.state[:2] - self.goal)

        # Reward the agent for moving closer to the goal.
        progress = self.previous_distance - current_distance
        reward = progress * 10.0

        # Small penalty to discourage wild actions.
        action_penalty = 0.01 * np.sum(np.square(action))
        reward -= action_penalty

        self.previous_distance = current_distance

        terminated = False
        success = False
        out_of_bounds = False

        if current_distance < 0.5:
            reward += 100.0
            terminated = True
            success = True

        if abs(x) > self.world_size or abs(y) > self.world_size:
            reward -= 50.0
            terminated = True
            out_of_bounds = True

        truncated = self.step_count >= self.max_steps

        info = {
            "distance_to_goal": current_distance,
            "success": success,
            "out_of_bounds": out_of_bounds
        }

        return self._get_obs(), reward, terminated, truncated, info