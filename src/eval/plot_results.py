import os

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = os.path.join("results", "plots")
CSV_PATH = os.path.join("results", "evaluation_results.csv")


EVALUATION_RESULTS = [
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


def make_results_dataframe() -> pd.DataFrame:
    """Create a DataFrame from the evaluation results."""
    return pd.DataFrame(EVALUATION_RESULTS)


def add_horizontal_value_labels(ax, values, suffix=""):
    """Add labels to the end of horizontal bars."""
    max_value = max(values) if values else 0

    for index, value in enumerate(values):
        label_offset = max_value * 0.01 if max_value > 0 else 1
        ax.text(
            value + label_offset,
            index,
            f"{value:.2f}{suffix}",
            va="center",
            fontsize=9,
        )


def save_average_reward_plot(df: pd.DataFrame) -> None:
    """Save a horizontal bar chart comparing average reward by agent."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.barh(df["Agent"], df["Average Reward"])
    ax.set_title("Average Reward by Agent")
    ax.set_xlabel("Average Reward")
    ax.invert_yaxis()

    add_horizontal_value_labels(ax, df["Average Reward"].tolist())

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "average_reward_comparison.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_success_vs_out_of_bounds_plot(df: pd.DataFrame) -> None:
    """Save a grouped horizontal bar chart comparing success and out-of-bounds rates."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    y_positions = list(range(len(df)))
    bar_height = 0.35

    success_positions = [position - bar_height / 2 for position in y_positions]
    out_of_bounds_positions = [position + bar_height / 2 for position in y_positions]

    ax.barh(
        success_positions,
        df["Success Rate"],
        height=bar_height,
        label="Success Rate",
    )
    ax.barh(
        out_of_bounds_positions,
        df["Out-of-Bounds Rate"],
        height=bar_height,
        label="Out-of-Bounds Rate",
    )

    ax.set_title("Success Rate vs. Out-of-Bounds Rate")
    ax.set_xlabel("Rate (%)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["Agent"])
    ax.invert_yaxis()
    ax.legend(loc="lower right")

    for index, value in enumerate(df["Success Rate"]):
        ax.text(value + 1, success_positions[index], f"{value:.0f}%", va="center", fontsize=8)

    for index, value in enumerate(df["Out-of-Bounds Rate"]):
        ax.text(value + 1, out_of_bounds_positions[index], f"{value:.0f}%", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "success_vs_out_of_bounds.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    """Generate evaluation result plots and save the raw data as a CSV file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = make_results_dataframe()
    df.to_csv(CSV_PATH, index=False)

    save_average_reward_plot(df)
    save_success_vs_out_of_bounds_plot(df)

    print("Saved CSV to results/evaluation_results.csv")
    print("Saved plots to results/plots/")
    print("- average_reward_comparison.png")
    print("- success_vs_out_of_bounds.png")


if __name__ == "__main__":
    main()