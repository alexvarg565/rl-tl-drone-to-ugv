
import numpy as np

from src.envs.drone_env import DroneEnv


def test_reset_returns_expected_state():
    env = DroneEnv()

    observation, info = env.reset(seed=42)

    expected = np.array(
        [0.0, 0.0, 0.0, 0.0, 8.0, 8.0],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(observation, expected)

    assert observation.shape == (6,)
    assert env.step_count == 0
    assert info == {}

    env.close()


def test_zero_action_keeps_drone_stationary():
    env = DroneEnv()
    env.reset(seed=42)

    action = np.array([0.0, 0.0], dtype=np.float32)

    observation, reward, terminated, truncated, info = env.step(action)

    np.testing.assert_array_equal(
        observation[:4],
        [0.0, 0.0, 0.0, 0.0]
    )

    assert env.step_count == 1
    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert not info["success"]

    env.close()


def test_drone_moves_with_acceleration():
    env = DroneEnv()
    env.reset(seed=42)

    action = np.array([1.0, 0.0], dtype=np.float32)

    observation, reward, terminated, truncated, info = env.step(action)

    expected_state = [0.01, 0.0, 0.1, 0.0]

    np.testing.assert_allclose(
        observation[:4],
        expected_state,
        rtol=1e-6,
        atol=1e-7,
    )

    assert reward > 0
    assert not terminated
    assert not truncated

    env.close()


def test_drone_reaches_goal():
    env = DroneEnv()
    env.reset(seed=42)

    # Place the drone close enough to satisfy the goal condition.
    env.state = np.array(
        [7.6, 8.0, 0.0, 0.0],
        dtype=np.float32
    )

    env.previous_distance = np.linalg.norm(
        env.state[:2] - env.goal
    )

    action = np.array([0.0, 0.0], dtype=np.float32)

    observation, reward, terminated, truncated, info = env.step(action)

    assert terminated
    assert not truncated
    assert info["success"]
    assert not info["out_of_bounds"]
    assert info["distance_to_goal"] < 0.5
    assert reward >= 100.0

    env.close()


def test_drone_out_of_bounds():
    env = DroneEnv()
    env.reset(seed=42)

    # Start near the right boundary with positive X velocity.
    env.state = np.array(
        [9.95, 0.0, 1.0, 0.0],
        dtype=np.float32
    )

    env.previous_distance = np.linalg.norm(
        env.state[:2] - env.goal
    )

    action = np.array([0.0, 0.0], dtype=np.float32)

    observation, reward, terminated, truncated, info = env.step(action)

    assert observation[0] > env.world_size
    assert terminated
    assert not truncated
    assert info["out_of_bounds"]
    assert not info["success"]
    assert reward < 0

    env.close()


def test_episode_truncates_at_max_steps():
    env = DroneEnv()
    env.reset(seed=42)

    action = np.array([0.0, 0.0], dtype=np.float32)

    for _ in range(env.max_steps - 1):
        observation, reward, terminated, truncated, info = env.step(action)

        assert not terminated
        assert not truncated

    observation, reward, terminated, truncated, info = env.step(action)

    assert env.step_count == env.max_steps
    assert truncated
    assert not terminated
    assert not info["success"]
    assert not info["out_of_bounds"]

    env.close()
