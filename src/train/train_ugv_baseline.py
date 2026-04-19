import os
from stable_baselines3 import PPO
from src.envs.ugv_env import UGVEnv


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = UGVEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/ppo_ugv_baseline/"
    )

    model.learn(total_timesteps=50000)
    model.save("models/ugv_ppo_baseline")


if __name__ == "__main__":
    main()