import os
from stable_baselines3 import PPO, SAC, TD3


def _check_model_exists(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Make sure you trained and saved the UAV model first."
        )


def load_drone_ppo_weights_into_ugv(drone_model_path: str, ugv_env):
    _check_model_exists(drone_model_path)

    drone_model = PPO.load(drone_model_path)

    ugv_model = PPO(
        "MlpPolicy",
        ugv_env,
        verbose=1,
        tensorboard_log="logs/ppo_transfer/"
    )

    ugv_model.policy.load_state_dict(
        drone_model.policy.state_dict(),
        strict=False
    )

    return ugv_model


def load_drone_sac_weights_into_ugv(drone_model_path: str, ugv_env):
    _check_model_exists(drone_model_path)

    drone_model = SAC.load(drone_model_path)

    ugv_model = SAC(
        "MlpPolicy",
        ugv_env,
        verbose=1,
        tensorboard_log="logs/sac_transfer/"
    )

    ugv_model.policy.load_state_dict(
        drone_model.policy.state_dict(),
        strict=False
    )

    return ugv_model


def load_drone_td3_weights_into_ugv(drone_model_path: str, ugv_env):
    _check_model_exists(drone_model_path)

    drone_model = TD3.load(drone_model_path)

    ugv_model = TD3(
        "MlpPolicy",
        ugv_env,
        verbose=1,
        tensorboard_log="logs/td3_transfer/"
    )

    ugv_model.policy.load_state_dict(
        drone_model.policy.state_dict(),
        strict=False
    )

    return ugv_model