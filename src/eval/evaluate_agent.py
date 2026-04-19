import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from src.envs.ugv_env import UGVEnv


def evaluate_model(model, env, episodes=10):
    rewards = []
    successes = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

        # crude success check
        if reward > 0:
            successes += 1

    return {
        "avg_reward": float(np.mean(rewards)),
        "success_rate": successes / episodes
    }


def main():
    env = UGVEnv()

    models = {
        "ppo_baseline": PPO.load("models/ugv_ppo_baseline.zip"),
        "ppo_transfer": PPO.load("models/ugv_ppo_transfer.zip"),
        "sac": SAC.load("models/ugv_sac.zip"),
        "td3": TD3.load("models/ugv_td3.zip"),
    }

    for name, model in models.items():
        results = evaluate_model(model, env)
        print(f"{name}: {results}")


if __name__ == "__main__":
    main()