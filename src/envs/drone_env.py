import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.world_size = 10.0
        self.max_steps = 200
        self.step_count = 0

        # x, y, vx, vy, goal_x, goal_y
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # acceleration in x and y
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.state = None
        self.goal = None

    def _get_obs(self):
        return np.array([
            self.state[0], self.state[1],
            self.state[2], self.state[3],
            self.goal[0], self.goal[1]
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.goal = np.array([8.0, 8.0], dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, -1.0, 1.0)
        ax, ay = action

        x, y, vx, vy = self.state

        vx += ax * 0.1
        vy += ay * 0.1
        x += vx * 0.1
        y += vy * 0.1

        self.state = np.array([x, y, vx, vy], dtype=np.float32)

        dist = np.linalg.norm(np.array([x, y]) - self.goal)
        reward = -dist

        terminated = False
        if dist < 0.5:
            reward += 100.0
            terminated = True

        out_of_bounds = abs(x) > self.world_size or abs(y) > self.world_size
        if out_of_bounds:
            reward -= 50.0
            terminated = True

        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}