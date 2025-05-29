# MazeGame: Dynamic MiniGrid Environment with Curriculum Learning

A reinforcement learning framework featuring dynamic maze environments with moving obstacles, expert agents, and imitation learning, built on the MiniGrid platform.

## Overview

This project implements a dynamic RL environment with procedural content generation, curriculum learning, and expert imitation. Key features include a custom MiniGrid environment, adaptive expert agents, and a training pipeline for both imitation and PPO agents.

## Installation

### Prerequisites
```bash
pip install gymnasium minigrid stable-baselines3 torch numpy matplotlib imageio tqdm
```

### Environment Setup
Python 3.8+ is required. All modules are compatible with Gymnasium and MiniGrid.

## Usage

Each script accepts parameters via argparse or internal config files.
Refer to inline comments or edit configuration values directly when needed.

### Run Dynamic Environment
```bash
python PCGtest.py
```
Launches manual control to observe moving obstacle dynamics.

### Collect Expert Trajectories
```bash
python expert_collector.py
```
Collects dynamic expert data; parameters configurable in script.

### Train Imitation Model
```bash
python imitation_rl_trainer.py
```
Trains a behavior cloning model from expert data.

### Train PPO Agent via Curriculum
```bash
python train_pcg_curriculum.py
```
Runs curriculum-based PPO training over 17 stages.

## Configuration

### Environment Parameters
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

### Curriculum Progression
```
Stage 1:  8x8,  1 room,  0 barriers, 0 lava
Stage 5: 12x12, 2 rooms, 1 barrier, 0 lava
Stage 10: 16x16, 2 rooms, 2 barriers, 2 lava
Stage 17: 18x18, 5 rooms, 2 barriers, 2 lava
```

## Expert Agent

### Safety Modes
- **Cautious**: Avoids 3+ cells from obstacles
- **Normal**: Avoids adjacent cells
- **Aggressive**: Avoids overlaps only
- **Direct**: Ignores obstacles

Uses dynamic danger evaluation and A* pathfinding with adaptive safety logic.

## Imitation Learning

### Network
- MiniGrid-optimized CNN feature extractor
- Policy head with deterministic/stochastic options

### Training Example
```python
trainer.train_imitation_learning(
    expert_filename="expert_data",
    epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    validation_split=0.2
)
```

## Example: PPO Inference
```python
from stable_baselines3 import PPO
model = PPO.load("./models/ppo_pcg_stage17.zip")
action, _ = model.predict(obs, deterministic=True)
```

## Outputs

- `expert_data/*.pkl`: Expert trajectories
- `models/*.pth` / `.zip`: Imitation and PPO model weights
- `*.png`, `*.gif`: Training curves, evaluation visualizations

## Evaluation Results

| Method              | Max Success Rate | Final Stage / Level | Generalization        |
|---------------------|------------------|----------------------|------------------------|
| PPO (Curriculum)    | 99% (Stage 1)     | 38% (Stage 17)       | Drops with complexity |
| Imitation Learning  | 51% (Level 1)     | 15% (Level 6)        | Moderate              |
| Expert Agent        | ~80% mid-stage    | Varies               | Stable across levels  |

## Performance Summary

- **Expert**: 30–80% success depending on complexity
- **Imitation**: ~90% accuracy; replicates expert strategies
- **Curriculum PPO**: >80% success on complex stages

## Troubleshooting

| Issue                     | Fix                                        |
|--------------------------|---------------------------------------------|
| Data collection fails     | Reduce difficulty / adjust expert strategy |
| Imitation overfitting     | Use more data / regularize                 |
| PPO instability           | Tweak curriculum or hyperparameters        |
| Memory issues             | Lower batch size or num envs               |

## License
MIT License

```
