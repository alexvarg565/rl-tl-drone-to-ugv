import os
from stable_baselines3 import PPO
from src.envs.drone_env import DroneEnv


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = DroneEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/ppo_drone/"
    )

    model.learn(total_timesteps=50000)
    model.save("models/drone_ppo")


if __name__ == "__main__":
    main()