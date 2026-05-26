import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join("results", "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)

data = [
    {
        "Agent": "PPO Baseline",
        "Algorithm": "PPO",
        "Training Type": "Baseline",
        "Average Reward": 207.75,
        "Success Rate": 100.00,
        "Avg Episode Length": 65.00,
        "Out-of-Bounds Rate": 0.00,
    },
    {
        "Agent": "PPO Transfer",
        "Algorithm": "PPO",
        "Training Type": "Transfer",
        "Average Reward": 207.67,
        "Success Rate": 100.00,
        "Avg Episode Length": 65.00,
        "Out-of-Bounds Rate": 0.00,
    },
    {
        "Agent": "SAC Baseline",
        "Algorithm": "SAC",
        "Training Type": "Baseline",
        "Average Reward": 208.04,
        "Success Rate": 100.00,
        "Avg Episode Length": 66.00,
        "Out-of-Bounds Rate": 0.00,
    },
    {
        "Agent": "SAC Transfer",
        "Algorithm": "SAC",
        "Training Type": "Transfer",
        "Average Reward": 208.13,
        "Success Rate": 100.00,
        "Avg Episode Length": 69.00,
        "Out-of-Bounds Rate": 0.00,
    },
    {
        "Agent": "TD3 Baseline",
        "Algorithm": "TD3",
        "Training Type": "Baseline",
        "Average Reward": 37.58,
        "Success Rate": 0.00,
        "Avg Episode Length": 76.00,
        "Out-of-Bounds Rate": 100.00,
    },
    {
        "Agent": "TD3 Transfer",
        "Algorithm": "TD3",
        "Training Type": "Transfer",
        "Average Reward": 37.58,
        "Success Rate": 0.00,
        "Avg Episode Length": 76.00,
        "Out-of-Bounds Rate": 100.00,
    },
]

df = pd.DataFrame(data)


def add_value_labels(ax, values, suffix=""):
    for i, value in enumerate(values):
        ax.text(
            value + max(values) * 0.01 if max(values) > 0 else value + 1,
            i,
            f"{value:.2f}{suffix}",
            va="center",
            fontsize=9,
        )


def save_horizontal_bar(column_name, filename, title, xlabel, suffix=""):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(df["Agent"], df[column_name])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    add_value_labels(ax, df[column_name].tolist(), suffix)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=200)
    plt.close()


def save_success_vs_oob():
    comparison_df = df[["Agent", "Success Rate", "Out-of-Bounds Rate"]].copy()

    fig, ax = plt.subplots(figsize=(8, 4.5))

    y_positions = range(len(comparison_df))
    bar_height = 0.35

    success_positions = [y - bar_height / 2 for y in y_positions]
    oob_positions = [y + bar_height / 2 for y in y_positions]

    ax.barh(success_positions, comparison_df["Success Rate"], height=bar_height, label="Success Rate")
    ax.barh(oob_positions, comparison_df["Out-of-Bounds Rate"], height=bar_height, label="Out-of-Bounds Rate")

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(comparison_df["Agent"])
    ax.invert_yaxis()
    ax.set_xlabel("Rate (%)")
    ax.set_title("Success Rate vs Out-of-Bounds Rate")
    ax.legend()

    for i, value in enumerate(comparison_df["Success Rate"]):
        ax.text(value + 1, success_positions[i], f"{value:.0f}%", va="center", fontsize=8)

    for i, value in enumerate(comparison_df["Out-of-Bounds Rate"]):
        ax.text(value + 1, oob_positions[i], f"{value:.0f}%", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "success_vs_out_of_bounds.png"), dpi=200)
    plt.close()


save_horizontal_bar(
    "Average Reward",
    "average_reward_comparison.png",
    "Average Reward by Agent",
    "Average Reward",
)

save_success_vs_oob()

save_horizontal_bar(
    "Avg Episode Length",
    "episode_length_comparison.png",
    "Average Episode Length by Agent",
    "Average Episode Length",
)

df.to_csv(os.path.join("results", "evaluation_results.csv"), index=False)

print("Saved plots to results/plots/")
print("Saved CSV to results/evaluation_results.csv")