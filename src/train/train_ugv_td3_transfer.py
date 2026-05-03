import os
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from src.envs.ugv_env import UGVEnv
from src.models.transfer_utils import load_drone_td3_weights_into_ugv


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = UGVEnv()
    n_actions = env.action_space.shape[-1]

    # TD3 needs action noise during continued training/fine-tuning.
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    model = load_drone_td3_weights_into_ugv("models/drone_td3.zip", env)
    model.action_noise = action_noise

    try:
        model.learn(total_timesteps=50000)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current TD3 transfer model...")

    model.save("models/ugv_td3_transfer")
    print("Saved TD3 transfer model to models/ugv_td3_transfer.zip")


if __name__ == "__main__":
    main()