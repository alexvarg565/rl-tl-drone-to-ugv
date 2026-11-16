"""Immutable planar scenarios and geometry validation, independent of Gymnasium.

Coordinates are world-frame (x, y) pairs in meters. Positive rotation is
counterclockwise from +X. World bounds are closed intervals. Validation checks
geometry only; it does not establish that a vehicle can reach the goal.
"""

from dataclasses import dataclass
from math import dist, isfinite

from numpy.random import Generator


@dataclass(frozen=True)
class CircleObstacle:
    """A closed circular obstacle, with center and radius in meters."""

    center: tuple[float, float]
    radius: float

    def __post_init__(self):
        object.__setattr__(self, "center", tuple(float(v) for v in self.center))
        object.__setattr__(self, "radius", float(self.radius))


@dataclass(frozen=True)
class Scenario:
    """Realized world geometry; start and goal represent robot-center positions.

    Bounds are (minimum, maximum) pairs. Iterable inputs are copied to tuples
    so modifying caller-owned collections cannot alter this scenario.
    Use validate_scenario() to check geometry against a ScenarioConfig.
    """

    start: tuple[float, float]
    goal: tuple[float, float]
    x_bounds: tuple[float, float] = (-10.0, 10.0)
    y_bounds: tuple[float, float] = (-10.0, 10.0)
    obstacles: tuple[CircleObstacle, ...] = ()

    def __post_init__(self):
        for name in ("start", "goal", "x_bounds", "y_bounds"):
            object.__setattr__(self, name, tuple(float(v) for v in getattr(self, name)))
        object.__setattr__(self, "obstacles", tuple(self.obstacles))


@dataclass(frozen=True)
class ScenarioConfig:
    """Validation constraints in meters, separate from realized geometry.

    clearance is the robot footprint radius plus any safety margin. It applies
    to start/goal distance from walls and obstacle surfaces, not to spacing
    between obstacles. Use a clearance suitable for both vehicles when sharing
    scenarios. Minimum start-goal distance and clearance are inclusive limits.
    Sampling attempts are controlled separately by generate_scenario().
    """

    min_start_goal_distance: float = 1.0
    clearance: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "min_start_goal_distance", float(self.min_start_goal_distance))
        object.__setattr__(self, "clearance", float(self.clearance))


def _validate_pair(values: tuple[float, float], name: str) -> None:
    if len(values) != 2 or not all(isfinite(v) for v in values):
        raise ValueError(f"{name} must contain exactly two finite values")


def _validate_world(scenario: Scenario, config: ScenarioConfig) -> None:
    """Check constraints and fixed geometry independently of start and goal."""
    for name in ("min_start_goal_distance", "clearance"):
        value = getattr(config, name)
        if not isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    for name in ("x_bounds", "y_bounds"):
        bounds = getattr(scenario, name)
        _validate_pair(bounds, name)
        if bounds[0] >= bounds[1]:
            raise ValueError(f"{name} minimum must be less than maximum")

    for index, obstacle in enumerate(scenario.obstacles):
        label = f"obstacle[{index}]"
        if not isinstance(obstacle, CircleObstacle):
            raise ValueError(f"{label} must be a CircleObstacle")
        _validate_pair(obstacle.center, f"{label}.center")
        if not isfinite(obstacle.radius) or obstacle.radius <= 0:
            raise ValueError(f"{label}.radius must be finite and positive")

        for coordinate, bounds in zip(obstacle.center, (scenario.x_bounds, scenario.y_bounds)):
            if not bounds[0] <= coordinate <= bounds[1] or min(
                coordinate - bounds[0], bounds[1] - coordinate
            ) < obstacle.radius:
                raise ValueError(f"{label} must be fully contained within world boundaries")

    # Check relationships only after all individual obstacles have been validated.
    for index, obstacle in enumerate(scenario.obstacles):
        for other_index in range(index):
            other = scenario.obstacles[other_index]
            if dist(obstacle.center, other.center) <= obstacle.radius + other.radius:
                raise ValueError(f"obstacle[{other_index}] and obstacle[{index}] touch or overlap")


def validate_scenario(scenario: Scenario, config: ScenarioConfig) -> None:
    """Raise ValueError for invalid geometry or validation constraints.

    Start/goal positions must lie within the closed, clearance-inset world.
    Obstacles must be fully contained in the closed world (wall tangency is
    allowed), and must neither touch nor overlap one another. Start/goal
    positions must be strictly outside every obstacle and at least clearance
    meters from its surface: exact positive clearance is allowed, but contact
    with the surface is invalid even when clearance is zero.

    Comparisons use the supplied floating-point values without a tolerance.
    """
    _validate_world(scenario, config)

    for name in ("start", "goal"):
        position = getattr(scenario, name)
        _validate_pair(position, name)
        for coordinate, bounds in zip(position, (scenario.x_bounds, scenario.y_bounds)):
            if not bounds[0] <= coordinate <= bounds[1]:
                raise ValueError(f"{name} must be inside world boundaries")
            if min(coordinate - bounds[0], bounds[1] - coordinate) < config.clearance:
                raise ValueError(f"{name} does not meet boundary clearance")

    if dist(scenario.start, scenario.goal) < config.min_start_goal_distance:
        raise ValueError("start-to-goal distance is below min_start_goal_distance")

    for index, obstacle in enumerate(scenario.obstacles):
        for name in ("start", "goal"):
            distance = dist(getattr(scenario, name), obstacle.center)
            if distance <= obstacle.radius:
                raise ValueError(f"{name} touches or lies inside obstacle[{index}]")
            if distance - obstacle.radius < config.clearance:
                raise ValueError(f"{name} does not meet clearance from obstacle[{index}]")


def generate_scenario(
    rng: Generator,
    *,
    x_bounds: tuple[float, float] = (-10.0, 10.0),
    y_bounds: tuple[float, float] = (-10.0, 10.0),
    obstacles: tuple[CircleObstacle, ...] = (),
    config: ScenarioConfig = ScenarioConfig(),
    max_attempts: int = 1000,
) -> Scenario:
    """Sample start/goal pairs using only the supplied NumPy Generator.

    Each attempt draws both positions uniformly from the clearance-inset world
    and accepts the pair only if validate_scenario() succeeds. Obstacles remain
    fixed. The generator is advanced, never reseeded; equivalent initial RNG
    states and arguments reproduce the same scenario sequence.

    Invalid fixed geometry, constraints, or an empty inset raise ValueError
    before consuming randomness. max_attempts must be a positive Python int
    (not bool); rng must be a numpy.random.Generator. Exhausting the pair budget
    raises RuntimeError with the last rejection reason. Exhaustion alone does
    not prove the configuration is impossible or that a route exists.
    """
    if not isinstance(rng, Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive Python integer")

    world = Scenario(
        start=(0.0, 0.0), goal=(0.0, 0.0),
        x_bounds=x_bounds, y_bounds=y_bounds, obstacles=obstacles,
    )
    _validate_world(world, config)
    low = tuple(bounds[0] + config.clearance for bounds in (world.x_bounds, world.y_bounds))
    high = tuple(bounds[1] - config.clearance for bounds in (world.x_bounds, world.y_bounds))
    if any(lower > upper for lower, upper in zip(low, high)):
        raise ValueError("clearance leaves no valid start/goal positions within world boundaries")

    last_rejection = ""
    for _ in range(max_attempts):
        start, goal = rng.uniform(low=low, high=high, size=(2, 2))
        candidate = Scenario(
            start=start, goal=goal,
            x_bounds=world.x_bounds, y_bounds=world.y_bounds, obstacles=world.obstacles,
        )
        try:
            validate_scenario(candidate, config)
        except ValueError as error:
            last_rejection = str(error)
        else:
            return candidate

    raise RuntimeError(
        f"Could not generate a valid scenario after {max_attempts} attempts. "
        f"Last rejection: {last_rejection}"
    )
