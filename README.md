# RL_Spec_Ops

A custom reinforcement learning environment for tactical squad-based operations with multi-agent support and field-of-view mechanics.

## Overview

RL_Spec_Ops is a specialized reinforcement learning project that implements a custom tactical environment for training intelligent agents in squad-based operations. The project leverages cutting-edge RL frameworks including **PPO** (Proximal Policy Optimization) and **DQN** (Deep Q-Networks) for agent training.

## Features

- **Custom Tactical Environment**: Implemented with PettingZoo for multi-agent reinforcement learning
- **Field-of-View System**: Advanced FOV algorithms for realistic agent perception
- **Multiple Training Algorithms**: Support for PPO and DQN training strategies
- **Multi-Agent Support**: Train multiple agents simultaneously in shared environments
- **Visualization Tools**: Built-in visualization system for monitoring agent behavior
- **Map System**: Support for custom mission maps and scenarios

## Project Structure
RL_Spec_Ops/ 
├── custom-environment/ 
│ └── env/ 
│ ├── custom_environment.py # Main environment implementation 
│ ├── custom_fov_algo.py # Custom field-of-view algorithm 
│ ├── fov.py # FOV utilities 
│ ├── PPO.py # PPO training implementation 
│ ├── train_ppo.py # PPO training script 
│ ├── train_dqn.py # DQN training script 
│ ├── visualizer.py # Environment visualization 
│ ├── test.py # Testing utilities 
│ └── Maps/ # Custom mission maps 
├── requirements.txt 
└── README.md


## Requirements

- Python 3.8+
- PyTorch >= 1.13.1
- Ray[RLlib] == 2.8.0
- PettingZoo >= 1.24.0 (with Atari and Butterfly support)
- Gymnasium
- TensorBoard >= 2.11.2
- SuperSuit >= 3.9.0
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Bolt17803/RL_Spec_Ops.git
cd RL_Spec_Ops

2. Install dependencies:
bash
pip install -r requirements.txt

## Quick Start
### Training with PPO
bash
python custom-environment/env/train_ppo.py

### Training with DQN
bash
python custom-environment/env/train_dqn.py

### Running Tests
bash
python custom-environment/env/test.py

### Visualizing the Environment
bash
python custom-environment/env/visualizer.py

## Key Components
- custom_environment.py: Core environment logic implementing the tactical scenario
- PPO.py: Proximal Policy Optimization algorithm implementation
- custom_fov_algo.py: Specialized field-of-view calculation for agent perception
- visualizer.py: Real-time visualization and monitoring of agent performance
- trainer.py: Training orchestration utilities

## Technologies Used
- PyTorch: Deep learning framework for neural networks
- Ray RLlib: Distributed reinforcement learning library
- PettingZoo: Multi-agent RL API
- Gymnasium: Environment API standard
- TensorBoard: Training visualization and monitoring

## License
This project is open source and available on GitHub.
