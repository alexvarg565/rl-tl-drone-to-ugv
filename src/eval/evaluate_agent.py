import os
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from src.envs.ugv_env import UGVEnv


def evaluate_model(model, env, episodes=10):
    """
    Runs a trained model for a fixed number of episodes and reports:
    - average reward
    - success rate
    - average episode length
    - out-of-bounds rate
    """

    rewards = []
    episode_lengths = []
    successes = 0
    out_of_bounds_count = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        final_info = {}

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            final_info = info

        rewards.append(total_reward)
        episode_lengths.append(steps)

        if final_info.get("success", False):
            successes += 1

        if final_info.get("out_of_bounds", False):
            out_of_bounds_count += 1

    return {
        "avg_reward": float(np.mean(rewards)),
        "success_rate": float(successes / episodes),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "out_of_bounds_rate": float(out_of_bounds_count / episodes),
    }


def load_if_exists(name, model_class, path):
    """
    Loads a trained model only if the saved model file exists.
    This prevents the evaluator from crashing if some models
    have not been trained yet.
    """

    if not os.path.exists(path):
        print(f"Skipping {name}: missing {path}")
        return None

    return model_class.load(path)


def main():
    env = UGVEnv()

    model_specs = {
        "PPO UGV Baseline": (PPO, "models/ugv_ppo_baseline.zip"),
        "PPO UGV Transfer": (PPO, "models/ugv_ppo_transfer.zip"),

        "SAC UGV Baseline": (SAC, "models/ugv_sac.zip"),
        "SAC UGV Transfer": (SAC, "models/ugv_sac_transfer.zip"),

        "TD3 UGV Baseline": (TD3, "models/ugv_td3.zip"),
        "TD3 UGV Transfer": (TD3, "models/ugv_td3_transfer.zip"),
    }

    print("\nEvaluation Results")
    print("------------------")

    for name, (model_class, path) in model_specs.items():
        model = load_if_exists(name, model_class, path)

        if model is None:
            continue

        results = evaluate_model(model, env)

        print(f"\n{name}")
        print(f"  Average Reward:        {results['avg_reward']:.2f}")
        print(f"  Success Rate:          {results['success_rate']:.2%}")
        print(f"  Avg Episode Length:    {results['avg_episode_length']:.2f}")
        print(f"  Out-of-Bounds Rate:    {results['out_of_bounds_rate']:.2%}")


if __name__ == "__main__":
    main()