import os
from stable_baselines3 import SAC
from src.envs.ugv_env import UGVEnv


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = UGVEnv()

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/sac_ugv/"
    )

    model.learn(total_timesteps=50000)
    model.save("models/ugv_sac")


if __name__ == "__main__":
    main()