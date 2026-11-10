
import torch
import pytest

from stable_baselines3 import PPO, SAC, TD3

from src.envs.drone_env import DroneEnv
from src.envs.ugv_env import UGVEnv
from src.models.transfer_utils import (
    load_drone_ppo_weights_into_ugv, load_drone_sac_weights_into_ugv, load_drone_td3_weights_into_ugv
)


def test_ppo_policy_weights_transfer(tmp_path):
    # Create a source UAV model without training it.
    source_model = PPO(
        "MlpPolicy",
        DroneEnv(),
        verbose=0,
    )

    # Save it to a temporary location.
    model_path = tmp_path / "test_drone_ppo.zip"
    source_model.save(str(model_path))

    # Initialize a UGV model using the source weights.
    target_model = load_drone_ppo_weights_into_ugv(
        str(model_path),
        UGVEnv(),
    )

    # Compare the source and target policy parameters.
    source_weights = source_model.policy.state_dict()
    target_weights = target_model.policy.state_dict()

    assert source_weights.keys() == target_weights.keys()

    for name in source_weights:
        assert torch.equal(
            source_weights[name],
            target_weights[name],
        ), f"Weight mismatch: {name}"


def test_sac_policy_weights_transfer(tmp_path):
    source_env = DroneEnv()
    target_env = UGVEnv()

    source_model = SAC(
        "MlpPolicy",
        source_env,
        verbose=0,
    )

    model_path = tmp_path / "test_drone_sac.zip"
    source_model.save(str(model_path))

    target_model = load_drone_sac_weights_into_ugv(
        str(model_path),
        target_env,
    )

    source_weights = source_model.policy.state_dict()
    target_weights = target_model.policy.state_dict()

    assert source_weights.keys() == target_weights.keys()

    for name in source_weights:
        assert torch.equal(
            source_weights[name],
            target_weights[name],
        ), f"SAC weight mismatch: {name}"

    source_env.close()
    target_env.close()


def test_td3_policy_weights_transfer(tmp_path):
    source_env = DroneEnv()
    target_env = UGVEnv()

    # Create an untrained UAV TD3 model.
    source_model = TD3(
        "MlpPolicy",
        source_env,
        verbose=0,
    )

    # Save the source model temporarily.
    model_path = tmp_path / "test_drone_td3.zip"
    source_model.save(str(model_path))

    # Transfer its policy weights to a UGV model.
    target_model = load_drone_td3_weights_into_ugv(
        str(model_path),
        target_env,
    )

    source_weights = source_model.policy.state_dict()
    target_weights = target_model.policy.state_dict()

    # Verify all expected parameter names are present.
    assert source_weights.keys() == target_weights.keys()

    # Verify that every parameter was copied exactly.
    for name in source_weights:
        assert torch.equal(
            source_weights[name],
            target_weights[name],
        ), f"TD3 weight mismatch: {name}"

    source_env.close()
    target_env.close()


@pytest.mark.parametrize(
    "transfer_function",
    [
        load_drone_ppo_weights_into_ugv,
        load_drone_sac_weights_into_ugv,
        load_drone_td3_weights_into_ugv,
    ],
)
def test_transfer_rejects_missing_model(tmp_path, transfer_function):
    model_path = tmp_path / "missing_model.zip"
    env = UGVEnv()

    try:
        with pytest.raises(
            FileNotFoundError,
            match="Model not found",
        ):
            transfer_function(str(model_path), env)
    finally:
        env.close()
