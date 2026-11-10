import numpy as np

from src.envs.ugv_env import UGVEnv


def test_reset_returns_expected_observation():
    env = UGVEnv()
    observation, info = env.reset(seed=42)

    np.testing.assert_array_equal(
        observation,
        np.array([0, 0, 0, 0, 8, 8], dtype=np.float32),
    )
    assert observation.shape == (6,)
    assert env.observation_space.contains(observation)
    assert env.step_count == 0
    assert info == {}
    env.close()


def test_zero_action_keeps_ugv_stationary():
    env = UGVEnv()
    env.reset(seed=42)

    observation, reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.0], dtype=np.float32)
    )

    np.testing.assert_allclose(observation[:4], [0, 0, 0, 0])
    assert reward == 0.0
    assert env.step_count == 1
    assert not terminated and not truncated
    assert not info["success"]
    env.close()


def test_controls_update_velocity_and_position():
    env = UGVEnv()
    env.reset(seed=42)

    observation, _, terminated, truncated, _ = env.step(
        np.array([1.0, 1.0], dtype=np.float32)
    )

    np.testing.assert_allclose(
        observation[:4], [0.008, 0.005, 0.08, 0.05],
        rtol=1e-5, atol=1e-7,
    )
    assert not terminated and not truncated
    env.close()


def test_actions_outside_range_are_clipped():
    env_a, env_b = UGVEnv(), UGVEnv()
    env_a.reset(seed=42)
    env_b.reset(seed=42)

    obs_a, reward_a, *_ = env_a.step(
        np.array([4.0, -3.0], dtype=np.float32)
    )
    obs_b, reward_b, *_ = env_b.step(
        np.array([1.0, -1.0], dtype=np.float32)
    )

    np.testing.assert_allclose(obs_a, obs_b)
    np.testing.assert_allclose(reward_a, reward_b)
    env_a.close()
    env_b.close()


def test_goal_success_ends_episode():
    env = UGVEnv()
    env.reset(seed=42)
    env.state = np.array([7.6, 8.0, 0.0, 0.0], dtype=np.float32)
    env.previous_distance = np.linalg.norm(env.state[:2] - env.goal)

    _, reward, terminated, truncated, info = env.step(
        np.zeros(2, dtype=np.float32)
    )

    assert terminated and not truncated
    assert info["success"] and not info["out_of_bounds"]
    assert reward >= 100.0
    env.close()


def test_boundary_failure_ends_episode():
    env = UGVEnv()
    env.reset(seed=42)
    env.state = np.array([9.95, 0.0, 1.0, 0.0], dtype=np.float32)
    env.previous_distance = np.linalg.norm(env.state[:2] - env.goal)

    observation, _, terminated, truncated, info = env.step(
        np.zeros(2, dtype=np.float32)
    )

    assert observation[0] > env.world_size
    assert terminated and not truncated
    assert info["out_of_bounds"] and not info["success"]
    env.close()


def test_episode_truncates_at_step_limit():
    env = UGVEnv()
    env.reset(seed=42)
    action = np.zeros(2, dtype=np.float32)

    for _ in range(env.max_steps - 1):
        _, _, terminated, truncated, _ = env.step(action)
        assert not terminated and not truncated

    _, _, terminated, truncated, _ = env.step(action)
    assert env.step_count == env.max_steps
    assert truncated and not terminated
    env.close()
