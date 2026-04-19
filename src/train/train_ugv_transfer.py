import os
from src.envs.ugv_env import UGVEnv
from src.models.transfer_utils import load_drone_weights_into_ugv


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = UGVEnv()

    model = load_drone_weights_into_ugv("models/drone_ppo.zip", env)
    model.learn(total_timesteps=50000)
    model.save("models/ugv_ppo_transfer")


if __name__ == "__main__":
    main()