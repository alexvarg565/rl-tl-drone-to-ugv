"""Behavioral tests for the new UAV; historical environment tests stay intact."""

from dataclasses import replace
from math import dist, inf, nan

import gymnasium as gym
import numpy as np
import pytest

from src.envs.navigation import (
    CircleObstacle,
    DroneNavigationEnv,
    Scenario,
    ScenarioConfig,
    generate_scenario,
)


@pytest.fixture
def fixed_scenario():
    return Scenario(start=(0.0, 0.0), goal=(8.0, 8.0))


@pytest.fixture
def env(fixed_scenario):
    environment = DroneNavigationEnv()
    environment.reset(options={"scenario": fixed_scenario})
    yield environment
    environment.close()


@pytest.mark.parametrize("name", ["dt", "max_acceleration", "max_axis_speed", "goal_radius"])
@pytest.mark.parametrize("value", [0.0, -1.0, nan, inf, -inf, True])
def test_invalid_physical_settings(name, value):
    with pytest.raises(ValueError, match=f"{name} must be finite and positive"):
        DroneNavigationEnv(**{name: value})


@pytest.mark.parametrize("name", ["max_steps", "max_sampling_attempts"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_integer_limits(name, value):
    with pytest.raises(ValueError, match=f"{name} must be a positive Python integer"):
        DroneNavigationEnv(**{name: value})


@pytest.mark.parametrize("name", ["x_bounds", "y_bounds"])
@pytest.mark.parametrize("bounds", [(1.0,), (0.0, 1.0, 2.0), (nan, 1.0), (0.0, inf), (1.0, 1.0), (2.0, 1.0)])
def test_invalid_world_bounds(name, bounds):
    with pytest.raises(ValueError, match=name):
        DroneNavigationEnv(**{name: bounds})


@pytest.mark.parametrize(
    "config, message",
    [
        (ScenarioConfig(clearance=-1.0), "clearance"),
        (ScenarioConfig(clearance=nan), "clearance"),
        (ScenarioConfig(min_start_goal_distance=inf), "min_start_goal_distance"),
        (ScenarioConfig(min_start_goal_distance=-1.0), "min_start_goal_distance"),
        (ScenarioConfig(clearance=11.0), "no valid start/goal"),
    ],
)
def test_invalid_scenario_constraints(config, message):
    with pytest.raises(ValueError, match=message):
        DroneNavigationEnv(scenario_config=config)


def test_wrong_scenario_config_type():
    with pytest.raises(TypeError, match="ScenarioConfig"):
        DroneNavigationEnv(scenario_config={})


@pytest.mark.parametrize(
    "settings",
    [
        {"x_bounds": (-1e40, 1e40)},
        {"dt": 1e40},
        {"max_axis_speed": 1e40},
        {"max_acceleration": 1e308, "dt": 10.0},
    ],
)
def test_settings_cannot_overflow_observations_or_integration(settings):
    with pytest.raises(ValueError, match="finite"):
        DroneNavigationEnv(**settings)


def test_fixed_reset_initializes_state_and_info(fixed_scenario):
    env = DroneNavigationEnv()
    observation, info = env.reset(seed=42, options={"scenario": fixed_scenario})
    np.testing.assert_array_equal(observation, [0, 0, 0, 0, 8, 8])
    assert observation.dtype == np.float32
    assert observation.shape == (6,)
    assert env.observation_space.contains(observation)
    assert env.scenario is fixed_scenario
    assert info == {
        "distance_to_goal": dist((0, 0), (8, 8)),
        "success": False, "out_of_bounds": False, "step_count": 0,
    }
    observation[0] = 100.0
    assert env.state[0] == 0.0
    assert fixed_scenario.start == (0.0, 0.0)


def test_random_reset_uses_gymnasium_rng_and_shared_generator():
    config = ScenarioConfig(clearance=0.5, min_start_goal_distance=3.0)
    env = DroneNavigationEnv(scenario_config=config)
    reference = np.random.default_rng(42)
    first, _ = env.reset(seed=42)
    assert env.scenario == generate_scenario(reference, config=config)
    second, _ = env.reset()
    assert env.scenario == generate_scenario(reference, config=config)
    assert not np.array_equal(first, second)
    replay, _ = env.reset(seed=42)
    np.testing.assert_array_equal(first, replay)


def test_different_seeds_produce_different_resets():
    env = DroneNavigationEnv()
    first, _ = env.reset(seed=42)
    second, _ = env.reset(seed=43)
    assert not np.array_equal(first, second)


def test_equal_seed_and_actions_reproduce_rollout():
    first, second = DroneNavigationEnv(), DroneNavigationEnv()
    first.reset(seed=42)
    second.reset(seed=42)
    for action in ([1.0, -0.5], [0.0, 0.0], [-1.0, 0.25]):
        first_result, second_result = first.step(action), second.step(action)
        np.testing.assert_array_equal(first_result[0], second_result[0])
        assert first_result[1:] == second_result[1:]


def test_fixed_reset_does_not_consume_scenario_randomness(fixed_scenario):
    env = DroneNavigationEnv()
    env.reset(seed=42, options={"scenario": fixed_scenario})
    env.reset()
    assert env.scenario == generate_scenario(np.random.default_rng(42))


def test_generation_attempt_budget_is_forwarded():
    env = DroneNavigationEnv(
        scenario_config=ScenarioConfig(min_start_goal_distance=100.0),
        max_sampling_attempts=2,
    )
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        env.reset(seed=42)
    with pytest.raises(gym.error.ResetNeeded):
        env.step([0.0, 0.0])


@pytest.mark.parametrize("field", ["x_bounds", "y_bounds"])
def test_fixed_scenario_must_match_world(fixed_scenario, field):
    env = DroneNavigationEnv()
    with pytest.raises(ValueError, match="configured world bounds"):
        env.reset(options={"scenario": replace(fixed_scenario, **{field: (-9.0, 9.0)})})


@pytest.mark.parametrize("start", [(20.0, 0.0), (nan, 0.0), (7.9, 8.0)])
def test_fixed_scenario_is_validated(fixed_scenario, start):
    with pytest.raises(ValueError):
        DroneNavigationEnv().reset(options={"scenario": replace(fixed_scenario, start=start)})


def test_fixed_scenario_obeys_configured_clearance(fixed_scenario):
    env = DroneNavigationEnv(scenario_config=ScenarioConfig(clearance=1.0))
    with pytest.raises(ValueError, match="boundary clearance"):
        env.reset(options={"scenario": replace(fixed_scenario, start=(-9.5, 0.0))})


def test_nonempty_obstacles_are_explicitly_unsupported(fixed_scenario):
    scenario = replace(fixed_scenario, obstacles=(CircleObstacle((4.0, 4.0), 1.0),))
    with pytest.raises(ValueError, match="obstacles are unsupported"):
        DroneNavigationEnv().reset(options={"scenario": scenario})


def test_reset_options_are_checked():
    env = DroneNavigationEnv()
    with pytest.raises(ValueError, match="dictionary"):
        env.reset(options=[])
    with pytest.raises(ValueError, match="only 'scenario'"):
        env.reset(options={"scenerio": None})
    with pytest.raises(TypeError, match="must be a Scenario"):
        env.reset(options={"scenario": None})


def test_zero_action_preserves_rest(env):
    observation, reward, terminated, truncated, info = env.step([0.0, 0.0])
    np.testing.assert_array_equal(observation[:4], [0, 0, 0, 0])
    assert reward == 0.0
    assert not terminated and not truncated
    assert info["step_count"] == 1


def test_acceleration_then_velocity_then_position_and_progress_reward(env):
    before = env.previous_distance
    observation, reward, terminated, truncated, info = env.step([1.0, -0.5])
    np.testing.assert_allclose(observation[:4], [0.01, -0.005, 0.1, -0.05], atol=1e-8)
    expected_distance = dist((0.01, -0.005), (8.0, 8.0))
    assert info["distance_to_goal"] == pytest.approx(expected_distance)
    assert reward == pytest.approx(10 * (before - expected_distance) - 0.0125)
    assert not terminated and not truncated


def test_configurable_dt_and_acceleration(fixed_scenario):
    env = DroneNavigationEnv(dt=0.2, max_acceleration=3.0)
    env.reset(options={"scenario": fixed_scenario})
    observation, *_ = env.step([1.0, -1.0])
    np.testing.assert_allclose(observation[:4], [0.12, -0.12, 0.6, -0.6])
    observation, *_ = env.step([0.0, 0.0])
    np.testing.assert_allclose(observation[:4], [0.24, -0.24, 0.6, -0.6])


def test_per_axis_velocity_limits_apply_before_position(fixed_scenario):
    env = DroneNavigationEnv(max_axis_speed=0.15)
    env.reset(options={"scenario": fixed_scenario})
    env.step([1.0, -1.0])
    observation, *_ = env.step([1.0, -1.0])
    np.testing.assert_allclose(observation[:4], [0.025, -0.025, 0.15, -0.15])
    assert np.linalg.norm(env.state[2:]) > env.max_axis_speed


def test_out_of_range_finite_actions_are_clipped(fixed_scenario):
    first, second = DroneNavigationEnv(), DroneNavigationEnv()
    first.reset(options={"scenario": fixed_scenario})
    second.reset(options={"scenario": fixed_scenario})
    first_result = first.step([1e300, -1e300])
    second_result = second.step([1.0, -1.0])
    np.testing.assert_array_equal(first_result[0], second_result[0])
    assert first_result[1:] == second_result[1:]


@pytest.mark.parametrize(
    "action",
    [None, 0.0, [], [1.0], [0.0, 0.0, 0.0], [[0.0, 0.0]],
     [nan, 0.0], [0.0, inf], [-inf, 0.0], [1j, 0.0], ["1", "0"],
     [True, False], [[1.0], [0.0, 1.0]]],
)
def test_invalid_action_does_not_advance_episode(env, action):
    before = env.state.copy()
    with pytest.raises(ValueError, match="two finite real numbers"):
        env.step(action)
    np.testing.assert_array_equal(env.state, before)
    assert env.step_count == 0
    env.step([0.0, 0.0])


def test_goal_success_terminates_at_step_limit(fixed_scenario):
    scenario = replace(fixed_scenario, start=(6.99, 8.0))
    # Cross into the radius in a single step, using public dynamics settings.
    env = DroneNavigationEnv(dt=1.0, max_steps=1)
    env.reset(options={"scenario": scenario})
    _, reward, terminated, truncated, info = env.step([1.0, 0.0])
    assert terminated and not truncated
    assert info["success"] and not info["out_of_bounds"]
    assert reward == pytest.approx(109.99)
    with pytest.raises(gym.error.ResetNeeded):
        env.step([0.0, 0.0])
    env.reset(options={"scenario": fixed_scenario})
    assert env.step_count == 0


def test_exact_goal_radius_is_not_success():
    env = DroneNavigationEnv(scenario_config=ScenarioConfig(min_start_goal_distance=0.0))
    env.reset(options={"scenario": Scenario((0.0, 0.0), (0.5, 0.0))})
    _, _, terminated, _, info = env.step([0.0, 0.0])
    assert not terminated and not info["success"]


def test_start_already_at_goal_is_evaluated_on_first_step():
    env = DroneNavigationEnv(scenario_config=ScenarioConfig(min_start_goal_distance=0.0))
    _, info = env.reset(options={"scenario": Scenario((0.0, 0.0), (0.0, 0.0))})
    assert not info["success"]
    _, reward, terminated, _, info = env.step([0.0, 0.0])
    assert terminated and info["success"] and reward == 100.0


@pytest.mark.parametrize("start, action", [((10.0, 0.0), [1, 0]), ((-10.0, 0.0), [-1, 0]), ((0.0, 10.0), [0, 1]), ((0.0, -10.0), [0, -1])])
def test_boundary_failure_preserves_terminal_observation(start, action):
    env = DroneNavigationEnv(dt=1.0, max_acceleration=2.0, max_steps=1)
    env.reset(options={"scenario": Scenario(start, (0.0, 0.0))})
    observation, reward, terminated, truncated, info = env.step(action)
    assert terminated and not truncated
    assert info["out_of_bounds"] and not info["success"]
    assert reward == pytest.approx(-70.01)
    assert env.observation_space.contains(observation)
    assert np.max(np.abs(observation[:2])) == 12.0
    with pytest.raises(gym.error.ResetNeeded):
        env.step([0.0, 0.0])


def test_boundary_failure_precedes_goal_success():
    env = DroneNavigationEnv(goal_radius=2.0)
    env.reset(options={"scenario": Scenario((10.0, 0.0), (9.0, 0.0))})
    _, reward, terminated, _, info = env.step([1.0, 0.0])
    assert info["distance_to_goal"] < env.goal_radius
    assert terminated and info["out_of_bounds"] and not info["success"]
    assert reward == pytest.approx(-50.11)


def test_world_boundary_itself_is_allowed():
    env = DroneNavigationEnv()
    env.reset(options={"scenario": Scenario((10.0, 0.0), (0.0, 0.0))})
    _, _, terminated, _, info = env.step([0.0, 0.0])
    assert not terminated and not info["out_of_bounds"]


def test_step_limit_truncates_and_requires_reset(fixed_scenario):
    env = DroneNavigationEnv(max_steps=2)
    env.reset(options={"scenario": fixed_scenario})
    assert env.step([0, 0])[2:4] == (False, False)
    _, _, terminated, truncated, info = env.step([0, 0])
    assert not terminated and truncated
    assert info["step_count"] == 2
    with pytest.raises(gym.error.ResetNeeded):
        env.step([0, 0])


def test_requires_initial_reset():
    with pytest.raises(gym.error.ResetNeeded):
        DroneNavigationEnv().step([0.0, 0.0])


def test_failed_reset_disables_stepping(env, fixed_scenario):
    with pytest.raises(ValueError):
        env.reset(options={"scenario": replace(fixed_scenario, start=(20.0, 0.0))})
    with pytest.raises(gym.error.ResetNeeded):
        env.step([0.0, 0.0])


def test_reset_restores_episode_state(env, fixed_scenario):
    env.step([1.0, 1.0])
    observation, info = env.reset(options={"scenario": fixed_scenario})
    np.testing.assert_array_equal(observation[:4], [0, 0, 0, 0])
    assert info["step_count"] == 0
    assert env.previous_distance == dist(fixed_scenario.start, fixed_scenario.goal)


def test_asymmetric_world_has_fixed_finite_spaces_and_bounded_rollouts():
    env = DroneNavigationEnv(x_bounds=(10.1, 20.3), y_bounds=(-3.7, 2.9), max_steps=20)
    low, high = env.observation_space.low.copy(), env.observation_space.high.copy()
    assert np.all(np.isfinite(low)) and np.all(np.isfinite(high))
    assert env.action_space.shape == (2,)
    assert env.action_space.dtype == np.float32
    for seed in range(5):
        observation, _ = env.reset(seed=seed)
        assert env.observation_space.contains(observation)
        rng = np.random.default_rng(seed)
        for _ in range(env.max_steps):
            observation, _, terminated, truncated, _ = env.step(rng.uniform(-1, 1, 2))
            assert env.observation_space.contains(observation)
            if terminated or truncated:
                break
    np.testing.assert_array_equal(env.observation_space.low, low)
    np.testing.assert_array_equal(env.observation_space.high, high)
