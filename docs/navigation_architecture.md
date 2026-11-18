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
and random start/goal generation with fixed obstacles. Runtime collision
handling remains deferred. The scenario module uses NumPy for random sampling
and has no Gymnasium dependency; dataclass values remain immutable Python tuples
and floats. Phase 2.2 adds a Gymnasium environment in the navigation package.

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

### Phase 2.2: DroneNavigationEnv

`DroneNavigationEnv` is implemented in `src/envs/navigation/drone.py` and exported
by `src.envs.navigation`. It leaves the original `DroneEnv` and experiments intact.
It is a point-mass, obstacle-free environment with no rendering or sensor model.

Constructor settings are keyword-only:

| Setting | Default | Meaning |
|---|---|---|
| `x_bounds`, `y_bounds` | `(-10.0, 10.0)` | Closed world intervals in meters |
| `dt` | `0.1` | Seconds per step |
| `max_acceleration` | `1.0` | Acceleration limit per axis, m/s² |
| `max_axis_speed` | `2.0` | Speed limit on each axis, m/s |
| `goal_radius` | `0.5` | Strict goal-distance threshold, meters |
| `max_steps` | `200` | Episode step budget |
| `scenario_config` | `ScenarioConfig()` | Shared initialization constraints |
| `max_sampling_attempts` | `1000` | Start/goal pair sampling budget |

Bounds must be finite and strictly ordered. Physical settings must be finite and
positive, and budgets must be positive Python integers (not booleans). Constraints
must be valid and leave a nonempty clearance-inset rectangle. Impossible random
start/goal separation is reported by the generator on reset. Configurations must
also permit finite float32 observation bounds and finite acceleration increments.
Bounds and physical settings are intended to remain fixed for the instance.

Physical state is `[x, y, vx, vy]`, integrated in float64. Actions are two finite
real numbers with shape `(2,)`, representing normalized world-X/Y acceleration.
Invalid shapes, nonnumeric, complex, boolean, NaN, and infinite actions raise
`ValueError` without advancing the episode. Finite out-of-range actions are
clipped to `[-1, 1]`. With clipped action `u`, each step uses:

```text
acceleration = max_acceleration * u
velocity_next = clip(velocity + acceleration * dt, -max_axis_speed, max_axis_speed)
position_next = position + velocity_next * dt
```

Zero action preserves existing velocity. Limits apply independently to X and Y,
so maximum diagonal speed is `sqrt(2) * max_axis_speed`. There is no drag, gravity,
attitude, altitude, or wind model.

Observations are fresh float32 arrays `[x, y, vx, vy, goal_x, goal_y]`. Their
`Box` bounds use the configured world bounds for goals, per-axis speed limits for
velocities, and world bounds expanded by `max_axis_speed * dt` for positions.
This includes a terminal position one step outside the world without clipping
physical state. Bounds are rounded outward in float32 and remain fixed across
resets. Observations retain physical units rather than applying normalization.

`reset(seed=..., options=None)` calls Gymnasium's seeded reset, then passes
`self.np_random` directly to `generate_scenario()`. `reset()` continues the RNG
sequence; repeating the same seed replays it. Action-space sampling has its own
seed. The environment can replay a fixed scenario through
`reset(options={"scenario": scenario})`: it must pass shared validation, match
the configured world bounds, and contain no obstacles. Fixed replay consumes no
scenario-sampling randomness. Nonempty obstacles are rejected explicitly because
valid initial clearance alone cannot prevent later collisions.

Reset initializes zero velocity, step count zero, and the previous goal distance.
It returns observation and info. `step()` requires a successful reset before the
first step and after either termination or truncation. A failed reset also disables
stepping until a reset succeeds. Starts already inside the goal radius are
evaluated on the first step, allowing the shared scenario validation rules to stay
unchanged. Unknown reset option keys are rejected.

After movement, a position strictly outside the closed world terminates with
boundary failure. Otherwise, a goal distance strictly less than `goal_radius`
terminates with success. Boundary failure takes precedence over success, and
termination takes precedence over step-limit truncation. The step limit is an
external cutoff, not an intrinsic finite-horizon task. Clearance affects initial
sampling and validation only; runtime boundary checks use the point position.
Goal and boundary outcomes are evaluated at step endpoints; continuous goal
crossing detection is not implemented.

The initial reward preserves the original baseline coefficients:

```text
reward = 10 * (previous_distance - distance) - 0.01 * sum(clipped_action ** 2)
reward += 100 on success, or -= 50 on boundary failure
```

Returned info includes `distance_to_goal` in meters, `success`, `out_of_bounds`,
and `step_count`, including at reset. Changing `dt` changes the physical duration
of an episode and cumulative per-step penalties; future timestep experiments
should also review reward scaling and discount factors.

The six observation meanings match the historical UAV but do not guarantee
future UGV policy compatibility. Heading-based steering needs orientation state
and has different action semantics. Shared units and coordinate frames support
later representation-transfer experiments without requiring whole-policy copies.

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
