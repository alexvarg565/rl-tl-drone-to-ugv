import os
from stable_baselines3 import SAC
from src.envs.drone_env import DroneEnv


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = DroneEnv()

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/sac_drone/"
    )

    try:
        model.learn(total_timesteps=50000)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current SAC drone model...")

    model.save("models/drone_sac")
    print("Saved SAC drone model to models/drone_sac.zip")


if __name__ == "__main__":
    main()