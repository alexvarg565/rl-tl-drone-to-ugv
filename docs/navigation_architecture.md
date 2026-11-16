# Navigation System Architecture

## Purpose

Develop a configurable UAV and UGV navigation simulation for reinforcement learning and cross-domain transfer experiments.

The new system will support randomized scenarios, improved vehicle dynamics, obstacles, and automated evaluation.

The original UAV and UGV implementations will remain unchanged as historical experiment baselines.

## Design Principles

- Separate scenario generation from robot movement and reinforcement learning.
- Use shared code for genuinely common functionality.
- Keep vehicle-specific dynamics in separate environment implementations.
- Use reproducible random seeds.
- Follow the Gymnasium environment interface.
- Add automated tests for each new component.
- Keep the simulation simple enough to understand, validate, and extend.

## World Definition

The initial simulation uses a two-dimensional Cartesian coordinate system.

- X axis: Horizontal world position
- Y axis: Vertical world position
- Position units: Meters
- Time units: Seconds
- Default world boundaries: -10 to +10 on both axes
- Default simulation time step: 0.1 seconds
- Coordinate orientation: Positive rotation is counterclockwise from the positive X axis

World dimensions and simulation settings will be configurable.

## Scenario Generation

A scenario contains:

- World boundaries
- Robot starting position
- Goal position
- Obstacle positions and sizes

`Scenario` stores realized geometry. `ScenarioConfig` separately stores validation
constraints; it does not duplicate world bounds or include vehicle dynamics,
simulation timestep, rewards, or sensors. `CircleObstacle`, `Scenario`, and
`ScenarioConfig` are frozen dataclasses. Coordinates and collections are copied
to immutable tuples, and numeric geometry values are stored as floats.

### Checkpoint 2.1: Scenario Data and Validation

This checkpoint implements scenario data, `validate_scenario(scenario, config)`,
and random start/goal generation with fixed obstacles. Vehicle environments and
runtime collision handling remain deferred. The navigation package uses NumPy
for random sampling and has no Gymnasium dependency; dataclass values remain
immutable Python tuples and floats.

- Coordinates are world-frame `(x, y)` pairs in meters. Bounds are separate
  `(minimum, maximum)` pairs for X and Y; each minimum must be strictly smaller
  than its maximum. All coordinates, bounds, and constraint values must be finite.
- The rectangular world includes its boundaries. Start and goal represent robot
  centers and must be inside the world, at least `clearance` meters from each wall.
  Exact boundary clearance is accepted, including wall contact at zero clearance.
- `clearance` is nonnegative and represents robot footprint radius plus a safety
  margin. Paired UAV/UGV scenarios should use a value suitable for both vehicles.
- Circular obstacles have positive, finite radii and must be fully contained in
  the world. They may touch a world boundary. Any pair of obstacles that touches,
  overlaps, or contains one another is rejected. Robot clearance is not imposed
  between obstacles.
- Start and goal must be strictly outside all obstacles and at least `clearance`
  meters from their surfaces. Exact positive clearance is accepted; obstacle
  contact is rejected even with zero clearance.
- Euclidean start-to-goal distance must be at least `min_start_goal_distance`
  (default 1 meter). This constraint is finite and nonnegative. A zero minimum
  explicitly permits coincident start and goal positions.
- Geometry is checked against the supplied floating-point values without a
  tolerance. Invalid geometry or constraints raise descriptive `ValueError`s.
  Validation does not prove route connectivity or vehicle-specific reachability.

### Reproducible Start and Goal Generation

The public API is:

```python
generate_scenario(
    rng: numpy.random.Generator,
    *,
    x_bounds=(-10.0, 10.0),
    y_bounds=(-10.0, 10.0),
    obstacles=(),
    config=ScenarioConfig(),
    max_attempts=1000,
) -> Scenario
```

The caller supplies the RNG explicitly. Bounds and fixed obstacles describe the
world; `config` supplies the existing distance and clearance constraints.
`max_attempts` is a positive Python integer (booleans are rejected), counting
complete start/goal pairs rather than individual positions.

Each attempt samples both positions uniformly from the rectangle inset by
`config.clearance`. A zero-width inset is allowed, including a single-point
region when the minimum distance is zero. Every candidate must pass
`validate_scenario()` before being returned; rejected pairs are discarded in full.
Obstacles are copied into the scenario without changing their centers or radii.
Random obstacle placement is not implemented.

Malformed world geometry or constraints raise `ValueError` before consuming any
randomness. Clearance that leaves an empty sampling region is also rejected
immediately. Other configurations that cannot yield an accepted pair, including
unachievable minimum distances or obstacle-blocked regions, exhaust the budget
and raise `RuntimeError` reporting the attempt count and last rejection reason.
Budget exhaustion can also occur in a feasible but difficult world; it does not
prove impossibility. Accepted geometry still does not guarantee a navigable route.

```python
import numpy as np
from src.envs.navigation import ScenarioConfig, generate_scenario

rng = np.random.default_rng(42)
config = ScenarioConfig(min_start_goal_distance=3.0, clearance=0.5)
first = generate_scenario(rng, config=config)
second = generate_scenario(rng, config=config)

# A fresh equivalent RNG replays the sequence from the beginning.
replay_rng = np.random.default_rng(42)
assert first == generate_scenario(replay_rng, config=config)
assert second == generate_scenario(replay_rng, config=config)
```

The generator advances the supplied RNG and never seeds or uses NumPy's global
random state. Reproducibility requires the same initial RNG state/bit generator,
arguments, and implementation. Repeated calls with one RNG produce a continuing
sequence; constructing a fresh RNG with the same seed for each call repeats the
first scenario. Future Gymnasium resets can pass `self.np_random` directly.
Save realized evaluation scenarios and record NumPy versions and generator
settings for research replay, since seeds alone do not protect against changes
to the sampling algorithm or software environment.

## UAV Environment

The new UAV environment will initially model two-dimensional movement.

State variables include:

- X and Y position
- X and Y velocity

Actions control acceleration along the X and Y axes.

This is a simplified planar navigation model, not a complete three-dimensional flight simulation.

## UGV Environment

The new UGV environment will model a ground vehicle using heading-based movement.

State variables include:

- X and Y position
- Heading angle
- Forward speed

Actions control throttle and steering.

Unlike the original implementation, steering will change vehicle orientation instead of directly modifying Y velocity.

## Observations and Transfer Learning

The UAV and UGV will share navigation-task concepts such as goal location and obstacle distances, but their physical states and controls may differ.

Observation spaces will be explicitly defined, bounded, and tested.

The architecture will allow future research into transferring shared navigation representations without assuming that complete policies can be copied between vehicles with different action meanings.

## Episode Rules

An episode may end when:

- The robot reaches its destination
- The robot collides with an obstacle
- The robot leaves the world boundaries

Reaching the maximum step limit will truncate the episode.

Reward functions will encourage progress toward the goal while penalizing collisions and unnecessary control effort.

Exact reward values will be specified and tested during environment implementation.

## Development Stages

1. Architecture and interface design
2. Scenario generation
3. UAV environment
4. UGV environment
5. Obstacles and sensors
6. Visualization and validation
7. Integration, documentation, and review

## Future Research

Phase 3 will introduce new training experiments, multi-seed evaluations, baseline comparisons, and transfer-learning strategies.

The original experiment results will remain preserved.
