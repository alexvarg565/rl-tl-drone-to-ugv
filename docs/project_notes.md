# Project Notes

## Project Goal

Investigate whether reinforcement learning policies trained in a simulated UAV navigation environment can improve UGV navigation training through transfer learning.

The project compares three reinforcement learning algorithms:

- Proximal Policy Optimization (PPO)
- Soft Actor-Critic (SAC)
- Twin Delayed DDPG (TD3)

Each algorithm is evaluated using a UGV agent trained from scratch and a UGV agent initialized with weights from a previously trained UAV policy.

## Original Implementation

The project uses custom Gymnasium environments with simplified two-dimensional movement.

### UAV Environment

- Position and velocity-based navigation
- Continuous acceleration controls
- Fixed starting position and goal
- Distance-based reward shaping
- Goal-reaching and boundary termination conditions

### UGV Environment

- Simplified two-dimensional ground navigation
- Continuous throttle and steering inputs that independently modify X and Y velocity
- Fixed starting position and goal
- Distance-based reward shaping
- Goal-reaching and boundary termination conditions

Neither environment currently implements obstacles, realistic vehicle steering, or randomized navigation scenarios.

### Transfer Learning

The transfer-learning pipeline initializes UGV policies using parameters from previously trained UAV policies.

PPO, SAC, and TD3 use separate training and transfer scripts.

Policy weights are loaded using strict parameter matching to verify architectural compatibility.

## Phase 1 — Code Quality and Verification

The original experiments have been preserved while improving the repository's engineering quality.

Completed improvements include:

- Refactored evaluation visualizations
- Updated plotting code to read recorded results from CSV rather than hardcoded values
- Added automated UAV environment tests
- Added automated UGV environment tests
- Verified PPO, SAC, and TD3 policy-weight transfers
- Added missing-model error tests
- Enforced strict policy parameter loading
- Added development dependencies and documented the tested environment

The automated test suite currently contains 19 passing tests.

These tests verify implementation behavior but do not reproduce the original model-training results.

## Main Results

PPO and SAC successfully navigated to the goal in the original fixed evaluation scenario.

TD3 did not reach the goal in the recorded evaluation and instead terminated by leaving the environment boundaries.

Transfer-learning agents did not demonstrate a meaningful performance advantage over agents trained from scratch in the recorded evaluation.

Detailed results are available in [Evaluation Summary](../results/evaluation_summary.md).

## Interpretation

The original experiments demonstrate a working pipeline for training, transferring, and evaluating reinforcement learning policies across two simplified navigation environments.

However, the fixed evaluation conditions limit the conclusions that can be drawn.

The similar PPO and SAC baseline-versus-transfer results do not establish improved sample efficiency or generalization from transfer learning.

The TD3 results indicate that further investigation of training configuration, exploration, reward scaling, and learned behavior is warranted.

These findings motivate more rigorous experiments using varied scenarios and multiple independent training seeds.

## Known Limitations

- Fixed starting conditions and goals
- No obstacles or collision avoidance
- Simplified UAV and UGV dynamics
- No realistic steering or vehicle heading
- No sensor noise or localization uncertainty
- Deterministic evaluation on a single navigation scenario
- No documented multi-seed training comparisons
- Historical training dependencies and random seeds not fully recorded

The project remains a simulation-based research prototype and is not intended for physical robot deployment.

## Future Development

### Phase 2 — Advanced Navigation Environments

- Introduce randomized starting positions and goals
- Implement obstacles and collision detection
- Improve the UGV movement model
- Add navigation visualization and trajectory recording
- Review observation spaces and reward functions

### Phase 3 — Improved RL Experiments

- Retrain PPO, SAC, and TD3 in the updated environments
- Evaluate across multiple independent random seeds
- Compare baseline, zero-shot transfer, and fine-tuned agents
- Measure training convergence and sample efficiency
- Test performance on unseen navigation scenarios
- Investigate TD3 training behavior and hyperparameters

### Long-Term Development

- Explore ROS 2 integration
- Introduce higher-fidelity robot simulation
- Develop demonstration videos
- Improve automated testing and continuous integration
- Expand experiment reproducibility

## Development Approach

Future development will prioritize small, testable changes with clear Git commits.

Major environment revisions will be evaluated separately from the original experiments so historical results remain preserved.