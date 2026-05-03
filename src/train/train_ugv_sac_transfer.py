import os
from src.envs.ugv_env import UGVEnv
from src.models.transfer_utils import load_drone_sac_weights_into_ugv


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = UGVEnv()

    model = load_drone_sac_weights_into_ugv("models/drone_sac.zip", env)

    try:
        model.learn(total_timesteps=50000)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current SAC transfer model...")

    model.save("models/ugv_sac_transfer")
    print("Saved SAC transfer model to models/ugv_sac_transfer.zip")


if __name__ == "__main__":
    main()