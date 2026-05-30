# Project Notes

## Goal

Compare baseline UGV reinforcement learning agents against UAV-to-UGV transfer-learning agents.

## Algorithms

- PPO
- SAC
- TD3

## Main Result

PPO and SAC achieved stable UGV navigation performance. TD3 failed in this setup and requires additional tuning.

## Interpretation

Transfer learning was implemented and evaluated, but in the current simplified environment it did not significantly outperform baseline training. More complex environments may better reveal transfer-learning advantages.

## Limitations

The environments are simplified simulations and do not fully model real UAV/UGV physics, sensor noise, or real-world deployment constraints.

## Future Work

- More complex obstacle layouts
- More realistic dynamics
- Hyperparameter sweeps
- Improved TD3 reward scaling
- Demo videos or environment visualization