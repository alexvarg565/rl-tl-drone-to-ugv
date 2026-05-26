# Cross-Domain Reinforcement Learning for UAV-to-UGV Autonomous Navigation

A reinforcement learning project comparing **PPO**, **SAC**, and **TD3** for autonomous navigation and testing whether policies trained in a UAV-style obstacle-navigation environment can transfer to a UGV navigation task.

The project includes custom Gymnasium environments, training scripts, saved model checkpoints, and evaluation tools for comparing baseline and transfer-learning agents.

---

## Overview

Training autonomous agents from scratch can be expensive and time-consuming, especially when each platform or environment requires a separate training process. This project explores whether learned behavior from one navigation domain can help accelerate or improve learning in another.

The experiment follows three stages:

1. Train UAV navigation agents.
2. Train baseline UGV agents from scratch.
3. Train UGV transfer agents using UAV-trained behavior as the starting point.

The goal is to compare baseline UGV performance against UAV-to-UGV transfer-learning performance across multiple reinforcement learning algorithms.

---

## Key Features

- Custom UAV and UGV navigation environments
- PPO, SAC, and TD3 training pipelines
- Baseline UGV training from scratch
- UAV-to-UGV transfer-learning experiments
- Saved model checkpoints for trained agents
- Evaluation script for comparing reward, success rate, episode length, and out-of-bounds behavior
- Reproducible Python project structure using Stable-Baselines3 and Gymnasium

---

## Algorithms

| Algorithm | Type | Why It Was Used |
|---|---|---|
| PPO | On-policy | Stable baseline for policy optimization and navigation tasks |
| SAC | Off-policy | Strong continuous-control algorithm with entropy-based exploration |
| TD3 | Off-policy | Continuous-control method using twin critics and delayed policy updates |

---

## Repository Structure

```text
rl-tl-drone-to-ugv/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── envs/
│   │   ├── drone_env.py
│   │   └── ugv_env.py
│   │
│   ├── train/
│   │   ├── train_drone_ppo.py
│   │   ├── train_drone_sac.py
│   │   ├── train_drone_td3.py
│   │   ├── train_ugv_baseline.py
│   │   ├── train_ugv_sac.py
│   │   ├── train_ugv_td3.py
│   │   ├── train_ugv_transfer.py
│   │   ├── train_ugv_sac_transfer.py
│   │   └── train_ugv_td3_transfer.py
│   │
│   ├── eval/
│   │   └── evaluate_agent.py
│   │
│   └── models/
│       └── transfer_utils.py
│
├── models/
│   ├── drone_ppo.zip
│   ├── drone_sac.zip
│   ├── drone_td3.zip
│   ├── ugv_ppo_baseline.zip
│   ├── ugv_ppo_transfer.zip
│   ├── ugv_sac.zip
│   ├── ugv_sac_transfer.zip
│   ├── ugv_td3.zip
│   └── ugv_td3_transfer.zip
│
├── results/
├── configs/
├── docs/
└── tests/
```

---

## Environment Design

### UAV Environment

The UAV environment represents an aerial obstacle-navigation task. The agent learns to move through a simulated space, avoid obstacles, remain within environment boundaries, and reach a target location.

### UGV Environment

The UGV environment represents a ground-vehicle navigation task. The agent learns to navigate toward a goal while avoiding obstacles and minimizing invalid movement or out-of-bounds behavior.

Both environments were designed to support comparison between agents trained from scratch and agents trained using transfer learning.

---

## Training Workflow

The project uses three training phases:

### 1. UAV Training

Train PPO, SAC, and TD3 agents in the UAV navigation environment.

### 2. UGV Baseline Training

Train UGV agents from scratch to create baseline performance results.

### 3. UAV-to-UGV Transfer Training

Use UAV-trained models or learned behavior as the starting point for UGV training.

The full experiment ran for more than **2 million simulation timesteps** across multiple training configurations.

---

## Evaluation Metrics

Agents were evaluated using:

- Average reward
- Success rate
- Average episode length
- Out-of-bounds rate
- Baseline vs. transfer-learning performance
- Training stability and convergence behavior

These metrics were selected to measure both reward optimization and practical navigation behavior.

---

## Results Summary

PPO and SAC produced the most stable UGV navigation behavior in the tested environments. TD3 required additional tuning for more consistent navigation success.

| Agent | Training Type | Result Summary |
|---|---|---|
| PPO UGV Baseline | From scratch | Stable navigation performance |
| PPO UGV Transfer | UAV-to-UGV transfer | Stable navigation performance |
| SAC UGV Baseline | From scratch | Stable navigation performance |
| SAC UGV Transfer | UAV-to-UGV transfer | Stable navigation performance |
| TD3 UGV Baseline | From scratch | Required additional tuning |
| TD3 UGV Transfer | UAV-to-UGV transfer | Required additional tuning |

### Main Finding

PPO and SAC were more reliable in this navigation setup, while TD3 was more sensitive to reward scaling, exploration behavior, and hyperparameter tuning.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/alexvarg565/rl-tl-drone-to-ugv.git
cd rl-tl-drone-to-ugv
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment.

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training Commands

Train UAV agents:

```bash
python -m src.train.train_drone_ppo
python -m src.train.train_drone_sac
python -m src.train.train_drone_td3
```

Train baseline UGV agents:

```bash
python -m src.train.train_ugv_baseline
python -m src.train.train_ugv_sac
python -m src.train.train_ugv_td3
```

Train transfer-learning UGV agents:

```bash
python -m src.train.train_ugv_transfer
python -m src.train.train_ugv_sac_transfer
python -m src.train.train_ugv_td3_transfer
```

---

## Evaluation

Run the evaluation script:

```bash
python -m src.eval.evaluate_agent
```

The evaluation script compares trained agents using reward, success rate, average episode length, and out-of-bounds behavior.

---

## Technologies Used

- Python
- Stable-Baselines3
- Gymnasium
- PyTorch
- NumPy
- Pandas
- Matplotlib
- TensorBoard
- Docker
- Linux / HPC workflows

---

## Limitations

This project is a simulation-based research prototype. The environments are simplified and do not fully model real-world UAV or UGV dynamics, sensor noise, localization uncertainty, or physical constraints.

Future work would require more realistic simulation, stronger environment randomization, broader hyperparameter tuning, and additional validation before deployment to physical robotic systems.

---

## Future Improvements

- Add environment visualizations or demo videos
- Add training curves and evaluation plots to the `results/` folder
- Run larger hyperparameter sweeps
- Improve reward shaping and penalty scaling
- Add more realistic obstacle layouts
- Add sensor-style observations
- Compare additional transfer-learning strategies
- Test policies in higher-fidelity simulation environments

---

## Author

**Alexander Vargas**  
Computer Engineering Student  
California State University, San Bernardino  
GitHub: [alexvarg565](https://github.com/alexvarg565)