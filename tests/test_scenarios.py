from dataclasses import FrozenInstanceError, replace
from math import inf, nan, nextafter

import pytest

from src.envs.navigation import (
    CircleObstacle,
    Scenario,
    ScenarioConfig,
    validate_scenario,
)


@pytest.fixture
def scenario():
    return Scenario(
        start=(-4.0, 0.0),
        goal=(4.0, 0.0),
        obstacles=(CircleObstacle(center=(0.0, 0.0), radius=1.0),),
    )


def test_valid_scenario(scenario):
    assert validate_scenario(scenario, ScenarioConfig(clearance=0.5)) is None


def test_valid_scenario_without_obstacles():
    validate_scenario(Scenario(start=(0.0, 0.0), goal=(8.0, 8.0)), ScenarioConfig())


def test_rectangular_world_with_independent_bounds():
    scenario = Scenario(
        start=(1.0, -2.0), goal=(5.0, 2.0),
        x_bounds=(0.0, 6.0), y_bounds=(-3.0, 3.0),
    )
    validate_scenario(scenario, ScenarioConfig(clearance=1.0))


@pytest.mark.parametrize(
    "instance, field, value",
    [
        (CircleObstacle((0.0, 0.0), 1.0), "radius", 2.0),
        (Scenario((0.0, 0.0), (8.0, 8.0)), "start", (1.0, 1.0)),
        (ScenarioConfig(), "clearance", 1.0),
    ],
)
def test_data_classes_are_frozen(instance, field, value):
    with pytest.raises(FrozenInstanceError):
        setattr(instance, field, value)


def test_mutable_inputs_are_copied_to_immutable_values():
    start, goal, center = [0, 0], [8, 8], [4, 4]
    x_bounds, y_bounds = [-10, 10], [-10, 10]
    obstacles = [CircleObstacle(center, 1)]
    scenario = Scenario(start, goal, x_bounds, y_bounds, obstacles)

    for values in (start, goal, center, x_bounds, y_bounds):
        values[0] = 99
    obstacles.clear()

    assert scenario == Scenario(
        (0.0, 0.0), (8.0, 8.0),
        obstacles=(CircleObstacle((4.0, 4.0), 1.0),),
    )
    assert isinstance(scenario.start, tuple)
    assert isinstance(scenario.obstacles, tuple)
    assert isinstance(scenario.obstacles[0].center, tuple)
    assert isinstance(scenario.start[0], float)
    validate_scenario(scenario, ScenarioConfig())


@pytest.mark.parametrize("field", ["start", "goal", "x_bounds", "y_bounds"])
@pytest.mark.parametrize("value", [(nan, 1.0), (0.0, inf), (-inf, 1.0), (0.0,), (0.0, 1.0, 2.0)])
def test_invalid_coordinate_pairs(scenario, field, value):
    with pytest.raises(ValueError, match=f"{field} must contain exactly two finite values"):
        validate_scenario(replace(scenario, **{field: value}), ScenarioConfig())


@pytest.mark.parametrize("field", ["x_bounds", "y_bounds"])
@pytest.mark.parametrize("bounds", [(1.0, 1.0), (2.0, -2.0)])
def test_bounds_must_be_strictly_ordered(scenario, field, bounds):
    with pytest.raises(ValueError, match=f"{field} minimum must be less than maximum"):
        validate_scenario(replace(scenario, **{field: bounds}), ScenarioConfig())


@pytest.mark.parametrize("field", ["min_start_goal_distance", "clearance"])
@pytest.mark.parametrize("value", [-1.0, nan, inf, -inf])
def test_invalid_configuration(scenario, field, value):
    with pytest.raises(ValueError, match=f"{field} must be finite and nonnegative"):
        validate_scenario(scenario, ScenarioConfig(**{field: value}))


@pytest.mark.parametrize("field", ["start", "goal"])
@pytest.mark.parametrize("position", [(-10.01, 0.0), (10.01, 0.0), (0.0, -10.01), (0.0, 10.01)])
def test_positions_outside_world_are_rejected(scenario, field, position):
    with pytest.raises(ValueError, match=f"{field} must be inside world boundaries"):
        validate_scenario(replace(scenario, **{field: position}), ScenarioConfig())


@pytest.mark.parametrize("position", [(-10.0, -10.0), (10.0, -10.0), (-10.0, 10.0), (10.0, 10.0)])
def test_closed_world_boundary_is_valid_at_zero_clearance(position):
    validate_scenario(Scenario(position, (0.0, 0.0)), ScenarioConfig())


@pytest.mark.parametrize("field", ["start", "goal"])
def test_exact_boundary_clearance_is_valid(scenario, field):
    validate_scenario(
        replace(scenario, **{field: (-9.0, 9.0)}), ScenarioConfig(clearance=1.0)
    )


@pytest.mark.parametrize("field", ["start", "goal"])
def test_insufficient_boundary_clearance_is_rejected(scenario, field):
    with pytest.raises(ValueError, match=f"{field} does not meet boundary clearance"):
        validate_scenario(
            replace(scenario, **{field: (-9.5, 0.0)}), ScenarioConfig(clearance=1.0)
        )


def test_clearance_that_leaves_no_valid_position_is_rejected():
    with pytest.raises(ValueError, match="boundary clearance"):
        validate_scenario(Scenario((0.0, 0.0), (1.0, 0.0)), ScenarioConfig(clearance=11.0))


def test_minimum_distance_is_inclusive():
    scenario = Scenario((0.0, 0.0), (3.0, 4.0))
    validate_scenario(scenario, ScenarioConfig(min_start_goal_distance=5.0))
    with pytest.raises(ValueError, match="start-to-goal distance"):
        validate_scenario(scenario, ScenarioConfig(min_start_goal_distance=nextafter(5.0, inf)))


def test_coincident_start_and_goal_require_zero_minimum_distance():
    scenario = Scenario((0.0, 0.0), (0.0, 0.0))
    validate_scenario(scenario, ScenarioConfig(min_start_goal_distance=0.0))
    with pytest.raises(ValueError, match="start-to-goal distance"):
        validate_scenario(scenario, ScenarioConfig())


@pytest.mark.parametrize("radius", [0.0, -1.0, nan, inf, -inf])
def test_invalid_obstacle_radius(scenario, radius):
    with pytest.raises(ValueError, match=r"obstacle\[0\].radius must be finite and positive"):
        validate_scenario(
            replace(scenario, obstacles=(CircleObstacle((0.0, 0.0), radius),)), ScenarioConfig()
        )


@pytest.mark.parametrize("center", [(nan, 0.0), (0.0, inf), (-inf, 0.0), (0.0,), (0.0, 1.0, 2.0)])
def test_invalid_obstacle_center(scenario, center):
    with pytest.raises(ValueError, match=r"obstacle\[0\].center must contain exactly two finite values"):
        validate_scenario(
            replace(scenario, obstacles=(CircleObstacle(center, 1.0),)), ScenarioConfig()
        )


def test_obstacles_must_be_circle_obstacles(scenario):
    with pytest.raises(ValueError, match=r"obstacle\[0\] must be a CircleObstacle"):
        validate_scenario(replace(scenario, obstacles=((0.0, 0.0, 1.0),)), ScenarioConfig())


@pytest.mark.parametrize("center", [(-9.0, 0.0), (9.0, 0.0), (0.0, -9.0), (0.0, 9.0)])
def test_obstacle_wall_tangency_is_valid(scenario, center):
    validate_scenario(
        replace(scenario, obstacles=(CircleObstacle(center, 1.0),)), ScenarioConfig()
    )


@pytest.mark.parametrize("center", [(-9.1, 0.0), (9.1, 0.0), (0.0, -9.1), (0.0, 9.1), (11.0, 0.0)])
def test_obstacles_must_be_fully_contained(scenario, center):
    with pytest.raises(ValueError, match="fully contained"):
        validate_scenario(
            replace(scenario, obstacles=(CircleObstacle(center, 1.0),)), ScenarioConfig()
        )


@pytest.mark.parametrize("field", ["start", "goal"])
@pytest.mark.parametrize("position", [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)])
def test_obstacle_interior_and_surface_are_invalid_at_zero_clearance(scenario, field, position):
    with pytest.raises(ValueError, match=rf"{field} touches or lies inside obstacle\[0\]"):
        validate_scenario(replace(scenario, **{field: position}), ScenarioConfig())


@pytest.mark.parametrize("field", ["start", "goal"])
def test_exact_positive_obstacle_clearance_is_valid(scenario, field):
    validate_scenario(
        replace(scenario, **{field: (1.5, 0.0)}), ScenarioConfig(clearance=0.5)
    )


@pytest.mark.parametrize("field", ["start", "goal"])
def test_insufficient_obstacle_clearance_is_rejected(scenario, field):
    with pytest.raises(ValueError, match=rf"{field} does not meet clearance from obstacle\[0\]"):
        validate_scenario(
            replace(scenario, **{field: (nextafter(1.5, 0.0), 0.0)}),
            ScenarioConfig(clearance=0.5),
        )


@pytest.mark.parametrize("center", [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
def test_obstacle_overlap_and_tangency_are_rejected(scenario, center):
    obstacles = scenario.obstacles + (CircleObstacle(center, 1.0),)
    with pytest.raises(ValueError, match=r"obstacle\[0\] and obstacle\[1\] touch or overlap"):
        validate_scenario(replace(scenario, obstacles=obstacles), ScenarioConfig())


def test_separated_obstacles_do_not_require_robot_clearance_between_them(scenario):
    obstacles = scenario.obstacles + (CircleObstacle((0.0, 2.25), 1.0),)
    validate_scenario(replace(scenario, obstacles=obstacles), ScenarioConfig(clearance=0.5))


def test_nested_obstacles_are_rejected(scenario):
    obstacles = scenario.obstacles + (CircleObstacle((0.25, 0.0), 0.25),)
    with pytest.raises(ValueError, match="touch or overlap"):
        validate_scenario(replace(scenario, obstacles=obstacles), ScenarioConfig())
