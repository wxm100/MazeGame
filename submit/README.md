# MazeGame: Dynamic MiniGrid Environment with Curriculum Learning

A reinforcement learning framework featuring dynamic maze environments with moving obstacles, expert agents, and multiple learning strategies including PPO, imitation learning, and evolutionary algorithms. Built on the MiniGrid platform.

## Overview

This project implements a PCG-enabled RL environment with dynamic obstacle avoidance, curriculum learning, and expert-guided imitation. Key components include:

- A BSP-based MiniGrid environment with lava, balls, and doors
- Dynamic expert agent with flexible danger-aware pathfinding
- Curriculum-based PPO training
- Evolutionary strategy agent training
- Imitation learning from expert trajectories

## Installation

### Prerequisites
```bash
pip install gymnasium minigrid stable-baselines3 torch numpy matplotlib imageio tqdm
```

### Environment Setup
Python 3.8+ is required. Ensure all modules are compatible with Gymnasium 0.28+ and MiniGrid.

## Usage

### Manual Maze Test (PCG Environment)
```bash
python PCGtest.py
```
Launches interactive manual control in a procedurally generated environment.

### Expert Agent Test
```bash
python dynamic_expert_agent.py
```
Runs a safety-mode A* expert agent in a dynamic maze.

### Train PPO Agent (Curriculum Learning)
```bash
python train_ppo_curriculum.py
```
Trains a PPO agent across 17 curriculum stages with increasing complexity.

### Train EA Agent (Evolutionary Algorithm)
```bash
python EA_Train.py
```
Runs black-box optimization for training policy parameters using evolutionary strategies.

### Train Imitation Learning Agent
```bash
python imitation_rl_trainer.py
```
Trains a behavior cloning agent from pre-recorded expert trajectories.

## Configuration

### Sample Environment Parameters
```python
env_config = {
    'world_size': (18, 18),
    'room_count': 4,
    'goal_count': 1,
    'barrier_count': 6,
    'lava_count': 3,
    'lava_length': 4,
    'min_room_size': 5
}
```

### Curriculum Stages (`curriculum.py`)
```
Stage 1:  8x8,  1 room,  0 barrier, 0 lava
Stage 5: 12x12, 2 rooms, 1 barrier, 0 lava
Stage 10: 16x16, 3 rooms, 2 barriers, 1 lava
Stage 17: 18x18, 3 rooms, 4 barriers, 3 lava
```

## Expert Agent

The dynamic expert uses a progressive safety policy to plan safe paths under uncertainty:

- **Cautious**: Avoids cells within 3+ distance of balls
- **Normal**: Avoids adjacent tiles
- **Aggressive**: Only avoids direct collision
- **Direct**: Ignores obstacle presence

Pathfinding uses A*-based planning and emergency escape strategies.

## PPO Training

Curriculum-based PPO training uses a custom CNN encoder and environment wrappers. Each stage trains for 10M timesteps and is saved separately.

### PPO Inference Example
```python
from stable_baselines3 import PPO
model = PPO.load("models/ppo_pcg_stage17.zip")
action, _ = model.predict(obs, deterministic=True)
```

## Imitation Learning

Trains a policy to mimic expert trajectories using CNN-based feature extraction.

### Training Call Example
```python
trainer.train_imitation_learning(
    expert_filename="expert_data",
    epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    validation_split=0.2
)
```

## Evolutionary Algorithm (EA)

Trains agent behavior using black-box optimization without gradient backpropagation. Uses reward signals directly for population-based evolution.

## Outputs

- `expert_data/*.pkl`: Collected expert demonstrations
- `models/*.pth` / `.zip`: Saved models (EA, PPO, imitation)
- `*.png`, `*.gif`: Visualizations of policy execution

## Troubleshooting

| Issue                   | Fix or Suggestion                           |
|------------------------|---------------------------------------------|
| PPO training unstable   | Adjust curriculum pacing or hyperparams    |
| Expert fails to escape  | Lower stage complexity or increase retries |
| EA stagnates            | Use larger population or longer episodes   |
| Imitation overfits      | Add data, use dropout or validation split  |

## License

MIT License
