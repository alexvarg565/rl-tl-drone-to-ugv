import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from src.envs.drone_env import DroneEnv


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = DroneEnv()
    n_actions = env.action_space.shape[-1]

    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log="logs/td3_drone/"
    )

    try:
        model.learn(total_timesteps=50000)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current TD3 drone model...")

    model.save("models/drone_td3")
    print("Saved TD3 drone model to models/drone_td3.zip")


if __name__ == "__main__":
    main()