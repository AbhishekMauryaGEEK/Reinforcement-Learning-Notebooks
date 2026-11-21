# Pong Reinforcement Learning & Autonomous Control Suite

A comprehensive research benchmark comparing **Deterministic Kinematics**, **Tabular Q-Learning**, **Vector-State Deep Q-Networks (DQN)**, and **Visual Convolutional DQN (CNN-DQN)** on a custom, glassmorphic Pygame Pong simulation.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Simulation Environment & Physics Engine](#simulation-environment--physics-engine)
  - [Kinematics & Dynamic Deflection](#kinematics--dynamic-deflection)
  - [Markov Decision Process (MDP) Definition](#markov-decision-process-mdp-definition)
  - [Glassmorphic UI Engine](#glassmorphic-ui-engine)
- [Notebook 1: Model-Based Control via Deterministic Physics](#notebook-1-model-based-control-via-deterministic-physics)
  - [Analytical Ray-Folding Predictor](#analytical-ray-folding-predictor)
  - [Telemetry Logging System](#telemetry-logging-system)
  - [Empirical Validation & Prediction Accuracy](#empirical-validation--prediction-accuracy)
- [Notebook 2: Reinforcement Learning Paradigms](#notebook-2-reinforcement-learning-paradigms)
  - [1. Tabular Q-Learning (State Discretization)](#1-tabular-q-learning-state-discretization)
  - [2. Vector-State Deep Q-Network (DQN)](#2-vector-state-deep-q-network-dqn)
  - [3. Curriculum Learning & Opponent Scaling](#3-curriculum-learning--opponent-scaling)
  - [4. Visual Reinforcement Learning (CNN-DQN)](#4-visual-reinforcement-learning-cnn-dqn)
  - [5. Double Deep Q-Network (Double DQN)](#5-double-deep-q-network-double-dqn)
- [Comprehensive Benchmark Results](#comprehensive-benchmark-results)
- [Model Checkpoints & Telemetry Data](#model-checkpoints--telemetry-data)
  - [Pre-trained PyTorch Checkpoints](#pre-trained-pytorch-checkpoints)
  - [Gameplay Datasets](#gameplay-datasets)
- [Installation & Environment Setup](#installation--environment-setup)
- [Usage & Quickstart](#usage--quickstart)
  - [1. Running the Notebooks](#1-running-the-notebooks)
  - [2. Evaluating Pre-trained Checkpoints](#2-evaluating-pre-trained-checkpoints)
  - [3. Interactive Gameplay](#3-interactive-gameplay)
- [Core Theoretical Takeaways](#core-theoretical-takeaways)
- [License & Acknowledgments](#license--acknowledgments)

---

## Overview

This repository investigates the progression of autonomous game control from exact analytical modeling to deep policy optimization. Using a custom Pong environment built in Pygame, we benchmark four distinct control paradigms:

1. **Deterministic Physics Baseline (Geometric AI)**: An analytical kinematic solver using closed-form ray-folding reflection mechanics ($O(1)$ arithmetic complexity, near-zero error).
2. **Tabular Q-Learning**: Traditional reinforcement learning using spatial quantization (10-bin discretization per state dimension), highlighting the limits of discrete representations in continuous control.
3. **Vector-State Deep Q-Network (DQN)**: Deep reinforcement learning with continuous 6D state vectors, experience replay, and target network stabilization running on CUDA GPUs.
4. **Visual Reinforcement Learning (CNN-DQN)**: End-to-end pixel-to-action policy optimization trained directly on 4-frame temporal stacks of downsampled $84 \times 84$ grayscale game frames without hand-crafted features.

```
+------------------------------------------------------------------------------------------------+
|                                    CONTROL METHODOLOGY SPECTRUM                                |
|                                                                                                |
|   Analytical Kinematics          Tabular RL                Deep RL (MLP)       Visual Deep RL   |
|   +-------------------+      +----------------+         +------------------+   +--------------+ |
|   | Deterministic     | ---> | Discretized    | ------> | Continuous 6D    | ->| 4-Frame Pixel| |
|   | Ray-Folding       |      | Q-Table        |         | Vector DQN       |   | CNN-DQN      |
|   | ~100% Defense     |      | 0% Win Rate    |         | 67% Win Rate     |   | 18% Win Rate |
|   +-------------------+      +----------------+         +------------------+   +--------------+ |
+------------------------------------------------------------------------------------------------+
```

---

## Repository Structure

```
Ping pong RL/
│
├── PingPong-Model-based control using a deterministic physics model.ipynb
│   └── Model-based controller, ray-folding derivation, logging & trajectory validation.
│
├── PingPong-Reinforcement-Learning.ipynb
│   └── Full RL suite: Gym MDP, Tabular Q, Vector DQN, CNN-DQN, Double DQN & benchmarks.
│
├── pong_dqn_0.45_speed_100k.pt
│   └── Saved PyTorch checkpoint: Vector DQN trained for 100k steps against 0.45x opponent (65% win rate).
│
├── pong_dqn_0.50_speed_100k.pt
│   └── Saved PyTorch checkpoint: Vector DQN trained for 100k steps against 0.50x opponent (67% win rate).
│
├── pong_cnn_dqn_0.50_100k.pt
│   └── Saved PyTorch checkpoint: Visual CNN-DQN trained for 100k steps on 4-frame stacks (18% win rate).
│
├── pong_data.csv
│   └── Telemetry sample log containing frame-by-frame kinematics and impact angles (133 records).
│
├── pong_geometric_ai.csv
│   └── Extended gameplay telemetry dataset generated by the Geometric AI controller (3,918 records).
│
├── pong_data.xls
│   └── Large empirical dataset across 25 rallies (10,153 records) used for prediction validation.
│
├── a14f73fa325c54e372d60790441d7894.jpg
│   └── Glassmorphic theme background texture composited via multi-pass alpha blitting.
│
└── README.md
    └── Comprehensive technical documentation and benchmark report.
```

---

## Simulation Environment & Physics Engine

The custom environment models a continuous $800 \times 500$ pixel arena operating at 60 Hz.

| Parameter | Value | Description |
|---|---|---|
| `WIDTH` $\times$ `HEIGHT` | $800 \times 500\text{ px}$ | Arena resolution |
| `FPS` | $60\text{ Hz}$ | Target simulation frame rate |
| `BALL_SPEED` | $5.0\text{ px/frame}$ | Constant ball velocity magnitude |
| `PADDLE_SPEED` | $7.0\text{ px/frame}$ | Maximum vertical translation rate |
| `PADDLE_HEIGHT` $\times$ `WIDTH` | $100 \times 15\text{ px}$ | Paddle dimensions |
| `BALL_SIZE` | $12 \times 12\text{ px}$ | Bounding box of the square ball |
| `LEFT_PADDLE_X` | $30\text{ px}$ | Agent paddle horizontal position |
| `RIGHT_PADDLE_X` | $755\text{ px}$ (`WIDTH - 45`) | Opponent paddle horizontal position |

### Kinematics & Dynamic Deflection

1. **Top and Bottom Wall Reflection**: Elastic specular collision:
   $$v_y \leftarrow -v_y, \quad y \leftarrow \text{clamp}(y, 0, \text{HEIGHT} - \text{BALL\_SIZE})$$

2. **Paddle Deflection Dynamics**: The ball's deflection angle upon striking a paddle is dynamically computed based on the normalized contact distance from the paddle's vertical center:
   $$\text{relative\_hit} = \frac{\text{ball\_center}_y - \text{paddle\_center}_y}{\text{PADDLE\_HEIGHT} / 2} \in [-1.0, 1.0]$$
   $$\theta = \text{relative\_hit} \times 60^\circ \quad (\pm \frac{\pi}{3}\text{ rad})$$
   $$\vec{v}_{\text{ball}} = \begin{bmatrix} \pm \text{BALL\_SPEED} \cdot \cos(\theta) \\ \text{BALL\_SPEED} \cdot \sin(\theta) \end{bmatrix}$$
   This ensures that edge hits produce steep return angles, while center hits produce direct, horizontal trajectories.

### Markov Decision Process (MDP) Definition

The reinforcement learning agents interact with the custom `PongEnv` wrapper:

- **State Space ($\mathcal{S}$)**:
  - **Vector Observation**: $\mathbf{s}_t = [x_{\text{ball}}, y_{\text{ball}}, v_{x}, v_{y}, y_{\text{left}}, y_{\text{right}}] \in \mathbb{R}^6$
  - Normalized during DQN processing:
    $$\hat{\mathbf{s}}_t = \left[ \frac{x}{800}, \frac{y}{500}, \frac{v_x}{5}, \frac{v_y}{5}, \frac{y_{\text{left}}}{500}, \frac{y_{\text{right}}}{500} \right]$$
  - **Visual Observation**: Grayscale matrix downsampled to $84 \times 84 \times 1$, stacked temporally into $\mathbf{S}_t \in \mathbb{R}^{4 \times 84 \times 84}$ normalized to $[0.0, 1.0]$.
- **Action Space ($\mathcal{A}$)**: Discrete $\mathcal{A} = \{0, 1, 2\}$
  - `0`: `STAY` ($\Delta y = 0$)
  - `1`: `UP` ($\Delta y = -7\text{ px}$)
  - `2`: `DOWN` ($\Delta y = +7\text{ px}$)
- **Reward Function ($\mathcal{R}$)**:
  $$r_t = \begin{cases} +1.0 & \text{if agent scores (ball exceeds opponent backline } x > 800) \\ -1.0 & \text{if opponent scores (ball bypasses agent } x < -12) \\ +0.01 & \text{for every successful paddle return} \\ 0.0 & \text{otherwise} \end{cases}$$
- **Terminal Condition ($\mathcal{T}$)**: Point scored by either side, or maximum step horizon reached (timeout).

### Glassmorphic UI Engine

The simulation incorporates a modern UI rendered with Pygame:
- **Multi-pass Alpha Blitting**: Translucent frosted panels (`pygame.SRCALPHA`) with color `#FFFFFF` at $15\%$ opacity.
- **Dynamic Glow Lighting**: Radial gradient luminescence rendered around the ball and active paddle faces.
- **Custom Background Compositing**: High-resolution abstract glass texture (`a14f73fa325c54e372d60790441d7894.jpg`) blended with an interactive score HUD.

---

## Notebook 1: Model-Based Control via Deterministic Physics

The first notebook, `PingPong-Model-based control using a deterministic physics model.ipynb`, investigates the theoretical upper bound of performance using an analytical kinematic model.

### Analytical Ray-Folding Predictor

Instead of running step-by-step simulations, the Geometric AI uses a closed-form ray-folding algorithm to compute the ball's final $y$-coordinate when arriving at horizontal intercept $x_{\text{target}}$:

1. **Time-to-Intercept Calculation**:
   $$\Delta t = \frac{x_{\text{target}} - x_{\text{ball}}}{v_x}$$
2. **Unbounded Vertical Projection**:
   $$y_{\text{projected}} = y_{\text{ball}} + v_y \cdot \Delta t$$
3. **Folding Wall Bounces**:
   Because top ($y=0$) and bottom ($y=H_{\text{bound}}$) walls are specular reflectors, any overshoot beyond $[0, H_{\text{bound}}]$ is folded back into bounds:
   $$\begin{aligned}
   &\text{while } y_{\text{projected}} < 0 \text{ or } y_{\text{projected}} > H_{\text{bound}}: \\
   &\quad \text{if } y_{\text{projected}} < 0: \quad y_{\text{projected}} \leftarrow -y_{\text{projected}} \\
   &\quad \text{if } y_{\text{projected}} > H_{\text{bound}}: \quad y_{\text{projected}} \leftarrow 2 H_{\text{bound}} - y_{\text{projected}}
   \end{aligned}$$
4. **Proportional Actuation**:
   The paddle translates toward the predicted arrival point using a dead-zone tolerance window ($10\text{ px}$) to suppress high-frequency oscillation around the target coordinate.

### Telemetry Logging System

The notebook defines `PongLogger`, which serializes state snapshots into CSV format:
- `timestamp`: High-resolution datetime stamp
- `rally`, `frame`: Hierarchical temporal index
- `event`: Event tag (`hit`, `miss`, `wall_bounce`, `score`)
- `ball_x`, `ball_y`, `ball_vx`, `ball_vy`: Kinematic vectors
- `left_paddle_y`, `right_paddle_y`: Spatial positions of both competitors
- `hit_position`, `angle`: Contact offset and resulting launch angle
- `scorer`: Goal identifier

### Empirical Validation & Prediction Accuracy

Using the `pong_data.xls` telemetry dataset (10,153 logged frames across 25 rallies), the analytical predictor was evaluated against actual ground-truth impact positions:

```
+-----------------------------------------------------------------+
|               KINEMATIC PREDICTOR ACCURACY METRICS              |
+-----------------------------------------------------------------+
| Predictions Tested: 5,328 across 25 rallies                     |
| Mean Absolute Error:   1.285 pixels                             |
| Median Absolute Error: 0.061 pixels                             |
| Maximum Error:         8.459 pixels                             |
+-----------------------------------------------------------------+
```

> **Key Finding**: The median error of $0.061\text{ px}$ confirms that the geometric folding model is mathematically exact. The slight residual error in mean error ($1.285\text{ px}$) stems from discrete frame-boundary rounding during wall collisions.

---

## Notebook 2: Reinforcement Learning Paradigms

The second notebook, `PingPong-Reinforcement-Learning.ipynb`, establishes a comprehensive RL benchmark across multiple learning representations.

### 1. Tabular Q-Learning (State Discretization)

- **Discretization Schema**: Continuous 6D state mapped into 10 uniform buckets along each axis (effective bucket width: $80\text{ px}$ horizontally, $50\text{ px}$ vertically).
- **Update Rule**: 1-step Bellman Temporal Difference:
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
  with $\alpha = 0.1$, $\gamma = 0.99$, and $\epsilon$-greedy exploration decaying from $1.0$ to $0.01$.
- **Training Budget**: 5,000 episodes.
- **Evaluation**: **0.0% Win Rate** (0 wins, 100 losses, 0 timeouts).
- **Root Cause Analysis**:
  1. **State Aliasing**: Discretizing $800\text{ px}$ into 10 buckets means an entire $80\text{ px}$ region shares identical $Q$-values. Subtle ball angle variations are indistinguishable.
  2. **Credit Assignment Delay**: In a rally spanning $300+$ frames, propagating a sparse terminal reward ($+1 / -1$) backward without function approximation requires an astronomical number of iterations that cannot converge in 5,000 episodes.

### 2. Vector-State Deep Q-Network (DQN)

To overcome state aliasing, a Deep Q-Network parameterizes the action-value function over continuous normalized coordinates.

- **Network Architecture**:
  ```
  Input Layer (6 normalized features)
         │
  Dense(6 -> 128) + ReLU
         │
  Dense(128 -> 128) + ReLU
         │
  Dense(128 -> 3 Q-values)
  ```
- **Stabilization Mechanisms**:
  - **Replay Buffer**: Capacity $N = 100,000$ transitions, uniformly sampled with mini-batch size $64$.
  - **Target Network**: Frozen target network $\theta^-$ synchronized with online weights $\theta$ every $1,000$ optimization steps.
  - **Loss Function**: Mean Squared Error (MSE) on TD error:
    $$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$
  - **Optimizer**: Adam ($\alpha = 10^{-3}$, $\gamma = 0.99$).

### 3. Curriculum Learning & Opponent Scaling

The agent was benchmarked against scaled opponent speeds to test robustness and reaction capabilities:

- **0.45x Opponent Speed Benchmark (`pong_dqn_0.45_speed_100k.pt`)**:
  - Opponent Speed: $3.15\text{ px/frame}$ ($0.45 \times 7.0$), Reaction Noise: $\pm 40\text{ px}$.
  - Trained for $100,000$ steps.
  - **Evaluation (100 episodes)**: **65.0% Win Rate** (65 Wins, 8 Losses, 27 Timeouts).
- **0.50x Opponent Speed Benchmark (`pong_dqn_0.50_speed_100k.pt`)**:
  - Opponent Speed: $3.50\text{ px/frame}$ ($0.50 \times 7.0$), Reaction Noise: $\pm 40\text{ px}$.
  - Trained for $100,000$ steps.
  - **Evaluation (100 episodes)**: **67.0% Win Rate** (67 Wins, 12 Losses, 21 Timeouts).

```
DQN vs. 0.50x Opponent (100 Episodes):
  [========================================] 67% Wins
  [======] 12% Losses
  [===========] 21% Timeouts
```

### 4. Visual Reinforcement Learning (CNN-DQN)

To assess policy learning directly from visual perception without manual state extraction:

- **Input Preprocessing**:
  - Extract display surface pixels $\rightarrow$ convert to grayscale $\rightarrow$ downsample via area interpolation to $84 \times 84$.
  - Normalize pixel intensities to $[0.0, 1.0]$.
- **Temporal Stacking**:
  - Stack 4 consecutive frames: $\mathbf{X}_t \in \mathbb{R}^{4 \times 84 \times 84}$. Stacking provides the network with velocity, trajectory direction, and acceleration in a single observation.
- **DeepMind Nature Topology**:
  ```
  Input: (4, 84, 84)
         │
  Conv2d(in=4, out=32, kernel=8, stride=4) + ReLU    -> (32, 20, 20)
         │
  Conv2d(in=32, out=64, kernel=4, stride=2) + ReLU   -> (64, 9, 9)
         │
  Conv2d(in=64, out=64, kernel=3, stride=1) + ReLU   -> (64, 7, 7)
         │
  Flatten                                            -> (3136)
         │
  Dense(3136 -> 512) + ReLU
         │
  Dense(512 -> 3 Q-values)
  ```
- **Training Budget**: $100,000$ optimization steps with $5,000$-step warmup.
- **Evaluation (`pong_cnn_dqn_0.50_100k.pt`)**:
  - **18.0% Win Rate** (18 Wins, 65 Losses, 17 Timeouts).
  - Demonstrates successful visual feature acquisition; the lower win rate reflects the substantially higher sample complexity required to learn visual representations compared to compact 6D vectors.

### 5. Double Deep Q-Network (Double DQN)

Standard DQN suffers from maximization bias due to using $\max_{a'} Q(s', a'; \theta^-)$ for both action selection and evaluation. The notebook implements Double DQN logic:

$$Y_t^{\text{DoubleQ}} = r_{t+1} + \gamma Q\left( s_{t+1}, \arg\max_{a} Q(s_{t+1}, a; \theta_t); \theta_t^- \right)$$

By selecting the best action using online network $\theta_t$ and evaluating its value using target network $\theta_t^-$, overoptimistic value estimates are mitigated.

---

## Comprehensive Benchmark Results

The table below summarizes performance across all implemented paradigms evaluated under standardized conditions over 100 test episodes:

| Controller / Architecture | Input Space | Parameters | Compute Device | 100-Ep. Win Rate | Outcomes (W / L / T) | Core Finding |
|---|---|---|---|---|---|---|
| **Deterministic Geometric AI** | Analytical Kinematics | 0 (Closed form) | CPU | **~100.0%** | 100 / 0 / 0 | Exact reflection geometry; zero training sample requirement; $0.061\text{ px}$ median error. |
| **Tabular Q-Learning** | Discretized 6D (10 bins/axis) | $40,000$ table cells | CPU | **0.0%** | 0 / 100 / 0 | Failed due to state aliasing ($80\text{ px/bin}$) and sparse reward horizon. |
| **Vector DQN (0.45x Opponent)** | Normalized Continuous 6D | $\sim 34\text{k}$ weights | GPU (CUDA) | **65.0%** | 65 / 8 / 27 | Solves state aliasing; establishes stable continuous ball tracking. |
| **Vector DQN (0.50x Opponent)** | Normalized Continuous 6D | $\sim 34\text{k}$ weights | GPU (CUDA) | **67.0%** | 67 / 12 / 21 | Outperforms 0.45x baseline; capitalizes on opponent reaction latency. |
| **Visual CNN-DQN (0.50x Opponent)** | 4-Frame Stack $(4, 84, 84)$ | $\sim 1.68\text{M}$ weights | GPU (CUDA) | **18.0%** | 18 / 65 / 17 | Learns raw vision features; requires $>500\text{k}$ frames to reach vector parity. |

---

## Model Checkpoints & Telemetry Data

### Pre-trained PyTorch Checkpoints

The repository includes three fully trained PyTorch checkpoint files saved in standard dictionary format:

```python
import torch

# Load vector DQN checkpoint
ckpt = torch.load("pong_dqn_0.50_speed_100k.pt", map_location="cpu")
print("Training steps:", ckpt["training_steps"])
print("Evaluation win rate:", ckpt["evaluation_win_rate"])
```

| File Name | Size | Architecture | Checkpoint Contents |
|---|---|---|---|
| `pong_dqn_0.45_speed_100k.pt` | $295\text{ KB}$ | MLP (6 $\rightarrow$ 128 $\rightarrow$ 128 $\rightarrow$ 3) | `model_state_dict`, `target_state_dict`, `optimizer_state_dict`, hyperparameters & evaluation metadata |
| `pong_dqn_0.50_speed_100k.pt` | $148\text{ KB}$ | MLP (6 $\rightarrow$ 128 $\rightarrow$ 128 $\rightarrow$ 3) | `model_state_dict`, `target_state_dict`, `optimizer_state_dict`, hyperparameters & evaluation metadata |
| `pong_cnn_dqn_0.50_100k.pt` | $27.0\text{ MB}$ | Nature Conv2d + FC (512 $\rightarrow$ 3) | `model_state_dict`, `target_state_dict`, `optimizer_state_dict`, visual replay settings & evaluation metadata |

### Gameplay Datasets

| File Name | Format | Rows | Description |
|---|---|---|---|
| `pong_data.csv` | CSV | 133 | Small calibration and exploratory telemetry snippet containing hit angles and kinematics. |
| `pong_geometric_ai.csv` | CSV | 3,918 | Gameplay telemetry generated during human vs. Geometric AI matches. |
| `pong_data.xls` | Excel | 10,153 | Benchmark dataset of 25 full rallies used for statistical prediction error validation. |

**Telemetry Columns**:
`['timestamp', 'rally', 'frame', 'event', 'ball_x', 'ball_y', 'ball_vx', 'ball_vy', 'left_paddle_y', 'right_paddle_y', 'hit_position', 'angle', 'scorer']`

---

## Installation & Environment Setup

### Prerequisites
- Python 3.8+ (tested on Python 3.10 / 3.11)
- CUDA-compatible GPU recommended for Deep RL training (CPU execution is fully supported for inference)

### Setup Instructions

1. **Clone or navigate to the workspace**:
   ```bash
   cd "Ping pong RL"
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**:
   ```bash
   pip install torch torchvision numpy pygame pandas matplotlib openpyxl xlrd
   ```

4. **Verify GPU acceleration (Optional but recommended)**:
   ```bash
   python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

---

## Usage & Quickstart

### 1. Running the Notebooks

Launch JupyterLab or VS Code notebook interface:
```bash
jupyter lab
```

- Open `PingPong-Model-based control using a deterministic physics model.ipynb` to inspect the closed-form ray-folding mathematics, play against the geometric controller, and run trajectory error analyses.
- Open `PingPong-Reinforcement-Learning.ipynb` to step through the Gym environment, train the agents, inspect the visual frame stacker, and run comparative benchmarks.

### 2. Evaluating Pre-trained Checkpoints

To evaluate the best vector model (`pong_dqn_0.50_speed_100k.pt`) in Python:

```python
import torch
import numpy as np
from PingPong_RL import PongEnv, DQN  # or copy class definitions from notebook

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = DQN(state_size=6, action_size=3).to(device)
checkpoint = torch.load("pong_dqn_0.50_speed_100k.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Evaluation loop
env = PongEnv()
state = env.reset()
done = False
total_reward = 0

while not done:
    norm_state = np.array([
        state[0] / 800.0,
        state[1] / 500.0,
        state[2] / 5.0,
        state[3] / 5.0,
        state[4] / 500.0,
        state[5] / 500.0
    ], dtype=np.float32)

    state_tensor = torch.tensor(norm_state, device=device).unsqueeze(0)
    with torch.no_grad():
        action = torch.argmax(model(state_tensor), dim=1).item()

    state, reward, done = env.step(action)
    total_reward += reward

print("Episode finished. Total reward:", total_reward)
```

### 3. Interactive Gameplay

Both notebooks contain interactive Pygame loops allowing:
- **Human vs. Geometric AI**: Control the left paddle with `W`/`S` or `UP`/`DOWN` keys while the AI predicts and defends the right goal line.
- **AI vs. AI**: Watch two controllers (Geometric AI vs. Weak AI or DQN vs. Weak AI) rally in real-time.

---

## Core Theoretical Takeaways

1. **Analytical vs. Learning Duality**:
   When physical laws are deterministic and fully known, closed-form analytical models ($O(1)$ ray-folding) vastly outperform learning methods in compute efficiency, sample complexity, and defense rate ($0.061\text{ px}$ median error vs. thousands of training steps).
2. **The Pitfall of State Quantization**:
   Discretization of continuous physical spaces destroys gradient and velocity fidelity (*state aliasing*). Tabular Q-learning completely failed ($0.0\%$ win rate) because an $80\text{ px}$ bucket cannot distinguish critical deflection angles.
3. **Continuous Function Approximation**:
   DQN's multi-layer perceptron resolved state aliasing, achieving **$67.0\%$ win rate** by learning a continuous mapping from fine-grained coordinates to paddle actions.
4. **Sample Complexity in Visual RL**:
   While visual CNN-DQN eliminates the need for manual coordinate engineering, learning spatial representations directly from raw pixels requires significantly more training frames ($100\text{k}$ steps yielded $18.0\%$ win rate).
5. **Decoupled Action Evaluation**:
   Target networks and Double DQN formulations are vital in continuous dynamic environments to avoid systemic positive value estimation bias.

---

## License & Acknowledgments

- Built with [Pygame](https://www.pygame.org/) and [PyTorch](https://pytorch.org/).
- Reinforcement learning architectures based on classic Atari deep reinforcement learning paradigms (Mnih et al., 2013, 2015).
