import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join("results", "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)

data = [
    {
        "Agent": "PPO UGV Baseline",
        "Algorithm": "PPO",
        "Training Type": "Baseline",
        "Average Reward": 207.75,
        "Success Rate": 100.00,
        "Avg Episode Length": 65.00,
        "Out-of-Bounds Rate": 0.00,
    },
    {
        "Agent": "PPO UGV Transfer",
        "Algorithm": "PPO",
        "Training Type": "Transfer",
        "Average Reward": 207.67,
        "Success Rate": 100.00,
        "Avg Episode Length": 65.00,
        "Out-of-Bounds Rate": 0.00,
    },
    {
        "Agent": "SAC UGV Baseline",
        "Algorithm": "SAC",
        "Training Type": "Baseline",
        "Average Reward": 208.04,
        "Success Rate": 100.00,
        "Avg Episode Length": 66.00,
        "Out-of-Bounds Rate": 0.00,
    },
    {
        "Agent": "SAC UGV Transfer",
        "Algorithm": "SAC",
        "Training Type": "Transfer",
        "Average Reward": 208.13,
        "Success Rate": 100.00,
        "Avg Episode Length": 69.00,
        "Out-of-Bounds Rate": 0.00,
    },
    {
        "Agent": "TD3 UGV Baseline",
        "Algorithm": "TD3",
        "Training Type": "Baseline",
        "Average Reward": 37.58,
        "Success Rate": 0.00,
        "Avg Episode Length": 76.00,
        "Out-of-Bounds Rate": 100.00,
    },
    {
        "Agent": "TD3 UGV Transfer",
        "Algorithm": "TD3",
        "Training Type": "Transfer",
        "Average Reward": 37.58,
        "Success Rate": 0.00,
        "Avg Episode Length": 76.00,
        "Out-of-Bounds Rate": 100.00,
    },
]

df = pd.DataFrame(data)


def save_bar_chart(column_name, filename, title, ylabel):
    plt.figure(figsize=(10, 6))
    plt.bar(df["Agent"], df[column_name])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close()


save_bar_chart(
    "Average Reward",
    "average_reward_comparison.png",
    "Average Reward by Agent",
    "Average Reward",
)

save_bar_chart(
    "Success Rate",
    "success_rate_comparison.png",
    "Success Rate by Agent",
    "Success Rate (%)",
)

save_bar_chart(
    "Out-of-Bounds Rate",
    "out_of_bounds_rate_comparison.png",
    "Out-of-Bounds Rate by Agent",
    "Out-of-Bounds Rate (%)",
)

save_bar_chart(
    "Avg Episode Length",
    "episode_length_comparison.png",
    "Average Episode Length by Agent",
    "Average Episode Length",
)

csv_path = os.path.join("results", "evaluation_results.csv")
df.to_csv(csv_path, index=False)

print("Saved plots to results/plots/")
print("Saved CSV to results/evaluation_results.csv")