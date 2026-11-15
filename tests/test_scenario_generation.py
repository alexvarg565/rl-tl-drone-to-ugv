"""Randomized scenario behavior; existing validation tests remain unchanged."""

import numpy as np
import pytest

from src.envs.navigation import (
    CircleObstacle,
    Scenario,
    ScenarioConfig,
    generate_scenario,
    validate_scenario,
)


def test_equal_seeds_reproduce_a_sequence_without_reseeding():
    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)
    first = [generate_scenario(first_rng) for _ in range(4)]
    second = [generate_scenario(second_rng) for _ in range(4)]

    assert first == second
    assert len(set(first)) == 4


def test_different_seeds_produce_different_scenarios():
    assert generate_scenario(np.random.default_rng(42)) != generate_scenario(
        np.random.default_rng(43)
    )


def test_generation_does_not_consume_global_random_state():
    before = np.random.get_state()
    generate_scenario(np.random.default_rng(42))
    after = np.random.get_state()

    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("seed", [0, 1, 42, 123, 2026])
def test_generated_scenarios_obey_geometry_and_preserve_fixed_obstacles(seed):
    rng = np.random.default_rng(seed)
    obstacles = [CircleObstacle((12.0, -1.0), 0.75), CircleObstacle((17.0, 0.0), 1.0)]
    config = ScenarioConfig(min_start_goal_distance=3.0, clearance=0.5)

    for _ in range(25):
        scenario = generate_scenario(
            rng, x_bounds=(10.0, 20.0), y_bounds=(-4.0, 3.0),
            obstacles=obstacles, config=config,
        )

        validate_scenario(scenario, config)
        assert scenario.x_bounds == (10.0, 20.0)
        assert scenario.y_bounds == (-4.0, 3.0)
        assert scenario.obstacles == tuple(obstacles)
        assert np.linalg.norm(np.subtract(scenario.start, scenario.goal)) >= 3.0
        for position in (scenario.start, scenario.goal):
            assert 10.5 <= position[0] <= 19.5
            assert -3.5 <= position[1] <= 2.5
            for obstacle in obstacles:
                distance = np.linalg.norm(np.subtract(position, obstacle.center))
                assert distance - obstacle.radius >= config.clearance

    obstacles.clear()
    assert len(scenario.obstacles) == 2


def test_rejected_pair_is_retried():
    arguments = dict(
        x_bounds=(-2.0, 2.0), y_bounds=(-2.0, 2.0),
        obstacles=(CircleObstacle((0.0, 0.0), 1.0),),
        config=ScenarioConfig(clearance=0.25),
    )
    with pytest.raises(RuntimeError, match="after 1 attempts"):
        generate_scenario(np.random.default_rng(42), max_attempts=1, **arguments)

    scenario = generate_scenario(np.random.default_rng(42), max_attempts=500, **arguments)
    validate_scenario(scenario, arguments["config"])


def test_impossible_distance_stops_after_exact_attempt_budget():
    rng = np.random.default_rng(42)
    reference = np.random.default_rng(42)
    with pytest.raises(RuntimeError, match="after 3 attempts.*start-to-goal distance"):
        generate_scenario(
            rng, config=ScenarioConfig(min_start_goal_distance=100.0), max_attempts=3
        )

    # Three attempts consume three complete pairs, each with four coordinates.
    reference.uniform(-10.0, 10.0, size=(3, 2, 2))
    assert rng.bit_generator.state == reference.bit_generator.state


def test_obstacles_can_leave_no_valid_clearance_region():
    with pytest.raises(RuntimeError, match="after 5 attempts.*obstacle"):
        generate_scenario(
            np.random.default_rng(42),
            x_bounds=(-1.0, 1.0), y_bounds=(-1.0, 1.0),
            obstacles=(CircleObstacle((0.0, 0.0), 1.0),),
            config=ScenarioConfig(clearance=0.5, min_start_goal_distance=0.0),
            max_attempts=5,
        )


def test_clearance_can_reduce_sampling_region_to_a_single_point():
    scenario = generate_scenario(
        np.random.default_rng(42),
        x_bounds=(-1.0, 1.0), y_bounds=(-1.0, 1.0),
        config=ScenarioConfig(clearance=1.0, min_start_goal_distance=0.0),
        max_attempts=1,
    )
    assert scenario == Scenario(
        (0.0, 0.0), (0.0, 0.0), x_bounds=(-1.0, 1.0), y_bounds=(-1.0, 1.0)
    )


@pytest.mark.parametrize("max_attempts", [0, -1, 1.5, True, False, None])
def test_invalid_attempt_budget_is_rejected_without_sampling(max_attempts):
    rng = np.random.default_rng(42)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="max_attempts must be a positive Python integer"):
        generate_scenario(rng, max_attempts=max_attempts)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("rng", [None, 42, np.random.RandomState(42)])
def test_requires_an_explicit_numpy_generator(rng):
    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        generate_scenario(rng)


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"x_bounds": (1.0, 1.0)}, "minimum must be less than maximum"),
        ({"y_bounds": (0.0, np.inf)}, "two finite values"),
        ({"x_bounds": (0.0,)}, "two finite values"),
        ({"config": ScenarioConfig(clearance=-1.0)}, "clearance must be finite"),
        ({"config": ScenarioConfig(min_start_goal_distance=np.nan)}, "min_start_goal_distance"),
        ({"config": ScenarioConfig(clearance=11.0)}, "clearance leaves no valid"),
        ({"obstacles": (CircleObstacle((0.0, 0.0), 0.0),)}, "radius must be finite and positive"),
        ({"obstacles": (CircleObstacle((10.0, 0.0), 1.0),)}, "fully contained"),
        ({"obstacles": (CircleObstacle((0.0, 0.0), 1.0), CircleObstacle((2.0, 0.0), 1.0))}, "touch or overlap"),
    ],
)
def test_invalid_fixed_inputs_are_rejected_before_sampling(arguments, message):
    rng = np.random.default_rng(42)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match=message):
        generate_scenario(rng, **arguments)
    assert rng.bit_generator.state == before
