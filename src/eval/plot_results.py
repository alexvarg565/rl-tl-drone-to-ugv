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
    """Load previously recorded evaluation results from CSV."""

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(
            f"Evaluation results not found: {CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    required_columns = {
        "Agent",
        "Algorithm",
        "Training Type",
        "Average Reward",
        "Success Rate",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Evaluation results are empty.")

    return df


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


def save_baseline_vs_transfer_reward_plot(df: pd.DataFrame) -> None:
    """Save a grouped bar chart comparing baseline and transfer reward by algorithm."""
    pivot_df = df.pivot(
        index="Algorithm",
        columns="Training Type",
        values="Average Reward",
    ).loc[["PPO", "SAC", "TD3"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    pivot_df.plot(kind="bar", ax=ax)

    ax.set_title("Baseline vs. Transfer Average Reward")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Average Reward")
    ax.legend(title="Training Type")
    ax.tick_params(axis="x", rotation=0)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "baseline_vs_transfer_reward.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_success_rate_plot(df: pd.DataFrame) -> None:
    """Save a horizontal bar chart comparing success rate by agent."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.barh(df["Agent"], df["Success Rate"])
    ax.set_title("Success Rate by Agent")
    ax.set_xlabel("Success Rate (%)")
    ax.invert_yaxis()

    add_horizontal_value_labels(ax, df["Success Rate"].tolist(), suffix="%")

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "success_rate_comparison.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    """Generate evaluation result plots and save the raw data as a CSV file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = make_results_dataframe()

    save_average_reward_plot(df)
    save_baseline_vs_transfer_reward_plot(df)
    save_success_rate_plot(df)

    print("Saved plots to results/plots/")
    print("- average_reward_comparison.png")
    print("- baseline_vs_transfer_reward.png")
    print("- success_rate_comparison.png")


if __name__ == "__main__":
    main()