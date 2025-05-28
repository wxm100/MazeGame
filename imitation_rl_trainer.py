# imitation_trainer.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from PCGEnv import DynamicBSPMiniGridEnv
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio

# Add fixed dynamic environment class
class FixedDynamicBSPMiniGridEnv(DynamicBSPMiniGridEnv):
    """Fixed ball movement environment wrapper"""
    
    def step(self, action: int):
        # Manually execute ball movement logic
        for b in self.obstacles:
            ox, oy = b.cur_pos
            size_x, size_y = 3, 3
            top_x = min(max(ox-1, 1), self.width  - size_x - 1)
            top_y = min(max(oy-1, 1), self.height - size_y - 1)

            try:
                self.grid.set(ox, oy, None)
                newp = self.place_obj(
                    b,
                    top=(top_x, top_y),
                    size=(size_x, size_y),
                    max_tries=100
                )
                b.cur_pos = newp
            except (ValueError, AssertionError):
                ox = min(max(ox, 1), self.width-2)
                oy = min(max(oy, 1), self.height-2)
                b.cur_pos = (ox, oy)
                self.grid.set(ox, oy, b)

        obs, reward, term, trunc, info = super().step(action)
        
        # Check collision
        if action == self.actions.forward:
            front = self.grid.get(*self.front_pos)
            if front and front.type=="ball":
                term = True

        return obs, reward, term, trunc, info

class ExpertDataset(Dataset):
    """Expert dataset for imitation learning"""
    
    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        self.observations = torch.FloatTensor(observations)
        self.actions = torch.LongTensor(actions)
        
        # Transpose image dimensions (N, H, W, C) -> (N, C, H, W)
        if len(self.observations.shape) == 4:
            self.observations = self.observations.permute(0, 3, 1, 2)
        
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]

class MinigridSmallFeaturesExtractor(BaseFeaturesExtractor):
    """Features extractor for small MiniGrid observations"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        print(f"Observation space: {observation_space.shape}")
        
        # Handle observation space format uniformly
        if len(observation_space.shape) == 3:
            if observation_space.shape[0] == 3 or observation_space.shape[0] == 1:
                # PyTorch format: (C, H, W)
                n_input_channels, height, width = observation_space.shape
                print(f"Detected PyTorch format (C, H, W): C={n_input_channels}, H={height}, W={width}")
            else:
                # MiniGrid format: (H, W, C)
                height, width, n_input_channels = observation_space.shape
                print(f"Detected MiniGrid format (H, W, C): H={height}, W={width}, C={n_input_channels}")
        else:
            raise ValueError(f"Unsupported observation space shape: {observation_space.shape}")
        
        # CNN designed for small sizes
        if height <= 7 or width <= 7:
            # Small size version: use 1x1 convolution kernels
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, (1, 1)),
                nn.ReLU(),
                nn.Conv2d(16, 32, (1, 1)),  
                nn.ReLU(),
                nn.Flatten(),
            )
            print(f"Using small CNN architecture, input channels: {n_input_channels}")
        else:
            # Original version: use 2x2 convolution kernels
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 16, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                nn.Flatten(),
            )
            print(f"Using standard CNN architecture, input channels: {n_input_channels}")

        # Calculate flattened dimensions
        with torch.no_grad():
            # Create correct format sample input (PyTorch format: N, C, H, W)
            sample_input = torch.randn(1, n_input_channels, height, width)
            n_flatten = self.cnn(sample_input).shape[1]
            print(f"Flattened dimensions: {n_flatten}")

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class ImitationLearningNetwork(nn.Module):
    """Imitation learning network"""
    
    def __init__(self, observation_space, action_space, features_dim=128):
        super().__init__()
        
        # Use small-size adapted features extractor
        self.features_extractor = MinigridSmallFeaturesExtractor(
            observation_space, features_dim
        )
        
        # Policy head
        self.policy_head = nn.Linear(features_dim, action_space.n)
        
    def forward(self, observations):
        features = self.features_extractor(observations)
        logits = self.policy_head(features)
        return logits
    
    def predict_action(self, observation, deterministic=True):
        """Predict action"""
        with torch.no_grad():
            if len(observation.shape) == 3:
                observation = observation.unsqueeze(0)
            
            logits = self.forward(observation)
            
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)
                
            return action.cpu().numpy()

class ImitationTrainer:
    """Imitation learning trainer"""
    
    def __init__(self, env_config: Dict[str, Any], data_dir: str = "./expert_data"):
        self.env_config = env_config
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create environment to get observation and action spaces
        temp_env = self.create_single_env()
        self.observation_space = temp_env.observation_space
        self.action_space = temp_env.action_space
        temp_env.close()
        
        # Initialize network
        self.imitation_net = None
        
    def create_single_env(self, render_mode=None):
        """Create single environment - use fixed dynamic environment"""
        env = FixedDynamicBSPMiniGridEnv(
            render_mode=render_mode,
            **self.env_config
        )
        return ImgObsWrapper(env)
    
    def load_expert_data(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load expert data"""
        filepath = os.path.join(self.data_dir, f"{filename}.pkl")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Expert data file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        observations = data['observations']
        actions = data['actions']
        
        print(f"Original data:")
        print(f"  Observations: {len(observations)}")
        print(f"  Actions: {len(actions)}")
        print(f"  Observation shape: {observations.shape}")
        print(f"  Action range: {actions.min()} - {actions.max()}")
        
        # Fix data mismatch issue
        # Ensure observation and action counts match
        min_length = min(len(observations), len(actions))
        observations = observations[:min_length]
        actions = actions[:min_length]
        
        print(f"Corrected data:")
        print(f"  Observations: {len(observations)}")
        print(f"  Actions: {len(actions)}")
        print(f"  Used samples: {min_length}")
        
        return observations, actions
    
    def train_imitation_learning(self, 
                                expert_filename: str,
                                epochs: int = 20,
                                batch_size: int = 64,
                                learning_rate: float = 1e-3,
                                validation_split: float = 0.2,
                                save_path: str = "./models/imitation_model.pth"):
        """Train imitation learning model"""
        
        print("Starting imitation learning training")
        
        # Load expert data
        observations, actions = self.load_expert_data(expert_filename)
        
        # Split train and validation sets
        n_samples = len(actions)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val
        
        # Randomly shuffle data
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_dataset = ExpertDataset(observations[train_indices], actions[train_indices])
        val_dataset = ExpertDataset(observations[val_indices], actions[val_indices])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize network
        self.imitation_net = ImitationLearningNetwork(
            self.observation_space, self.action_space
        ).to(self.device)
        
        # Optimizer and loss function
        optimizer = optim.Adam(self.imitation_net.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training records
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        
        print(f"Train size: {n_train}, Val size: {n_val}")
        print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        for epoch in range(epochs):
            # Training phase
            self.imitation_net.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch_obs, batch_actions in train_pbar:
                batch_obs = batch_obs.to(self.device)
                batch_actions = batch_actions.to(self.device)
                
                optimizer.zero_grad()
                logits = self.imitation_net(batch_obs)
                loss = criterion(logits, batch_actions)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            self.imitation_net.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_obs, batch_actions in val_loader:
                    batch_obs = batch_obs.to(self.device)
                    batch_actions = batch_actions.to(self.device)
                    
                    logits = self.imitation_net(batch_obs)
                    loss = criterion(logits, batch_actions)
                    val_loss += loss.item()
                    
                    predictions = torch.argmax(logits, dim=-1)
                    correct += (predictions == batch_actions).sum().item()
                    total += batch_actions.size(0)
            
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_accuracy:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'model_state_dict': self.imitation_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'observation_space': self.observation_space,
                    'action_space': self.action_space
                }, save_path)
                print(f"  Saved best model to: {save_path}")
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses, val_accuracies)
        
        print("Imitation learning training completed!")
        return train_losses, val_losses, val_accuracies
    
    def evaluate_imitation_model(self, model_path: str, num_episodes: int = 10):
        """Evaluate imitation learning model"""
        
        print("Evaluating imitation learning model")
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.imitation_net = ImitationLearningNetwork(
            self.observation_space, self.action_space
        ).to(self.device)
        self.imitation_net.load_state_dict(checkpoint['model_state_dict'])
        self.imitation_net.eval()
        
        env = self.create_single_env(render_mode="rgb_array")
        
        success_count = 0
        episode_lengths = []
        frames_list = []
        collision_count = 0
        timeout_count = 0
        
        print(f"Starting evaluation in dynamic environment...")
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            frames = [env.render()]
            done = False
            steps = 0
            max_steps = 200
            
            print(f"Episode {episode+1}: Initial ball positions: {[ball.cur_pos for ball in env.unwrapped.obstacles]}")
            
            while not done and steps < max_steps:
                # Preprocess observation
                obs_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                # Predict action
                action = self.imitation_net.predict_action(obs_tensor, deterministic=True)[0]
                
                obs, reward, done, truncated, info = env.step(action)
                frames.append(env.render())
                steps += 1
                
                if done or truncated:
                    break
            
            episode_lengths.append(steps)
            if done and reward > 0:  # Success reaching goal
                success_count += 1
                print(f"  Success! Steps: {steps}")
            elif done and reward <= 0:
                collision_count += 1
                print(f"  Failed (possible collision). Steps: {steps}")
            else:
                timeout_count += 1
                print(f"  Timeout. Steps: {steps}")
            
            # Save first 5 episodes as GIF
            if episode < 5:
                frames_list.append(frames)
        
        env.close()
        
        success_rate = success_count / num_episodes
        avg_length = np.mean(episode_lengths)
        
        print(f"Imitation learning model evaluation results:")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Average steps: {avg_length:.1f}")
        print(f"Successful episodes: {success_count}/{num_episodes}")
        print(f"Collision failures: {collision_count}")
        print(f"Timeout failures: {timeout_count}")
        
        # Save evaluation GIFs
        for i, frames in enumerate(frames_list):
            imageio.mimsave(f"./models/imitation_eval_episode_{i+1}.gif", frames, fps=5)
        
        return success_rate, avg_length
    
    def plot_training_curves(self, train_losses, val_losses, val_accuracies):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        ax2.plot(val_accuracies, label='Val Accuracy', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('./models/training_curves.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """Main training pipeline"""
    
    # Environment configuration (consistent with dynamic data collection config)
    env_config = {
        'world_size': (18, 18),
        'room_count': 4,
        'goal_count': 1,
        'barrier_count': 6,  # Use dynamic obstacles
        'lava_count': 3,
        'lava_length': 4,
        'min_room_size': 5
    }
    
    # Create trainer - use default expert_data directory
    trainer = ImitationTrainer(env_config, data_dir="./expert_data")
    
    # Expert data filename (modify according to your actual filename)
    expert_data_filename = "simple_dynamic_expert_final_20000"  # Without .pkl extension
    
    print("Starting imitation learning pipeline")
    print(f"Environment config: {env_config}")
    print(f"Expert data file: {expert_data_filename}.pkl")
    print("Note: Using dynamic environment data for training")
    
    try:
        # Check if expert data file exists
        expert_file_path = os.path.join("./expert_data", f"{expert_data_filename}.pkl")
        if not os.path.exists(expert_file_path):
            print(f"Error: Expert data file not found {expert_file_path}")
            print("Please ensure dynamic expert data collection program has been run and data generated")
            print("Or modify expert_data_filename variable to correct filename")
            return
        
        # Train imitation learning model
        print("Training imitation learning model (learning dynamic avoidance behavior)")
        trainer.train_imitation_learning(
            expert_filename=expert_data_filename,
            epochs=20,
            batch_size=64,
            learning_rate=1e-3,
            save_path="./models/dynamic_imitation_model.pth"
        )
        
        # Evaluate imitation learning model
        print("Evaluating imitation learning model (dynamic environment)")
        success_rate, avg_length = trainer.evaluate_imitation_model(
            "./models/dynamic_imitation_model.pth",
            num_episodes=20
        )
        
        # Summary results
        print("Dynamic environment training completed! Results summary:")
        print(f"Environment config: {env_config}")
        print(f"Training data: Dynamic expert data (includes avoidance behavior)")
        print(f"Imitation learning model:")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average steps: {avg_length:.1f}")
        
        print(f"Model learned:")
        print(f"   - Dynamic obstacle avoidance")
        print(f"   - Expert behavioral cloning") 
        print(f"   - Safety-first strategy")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure expert data file exists, file path should be:")
        print(f"  ./expert_data/{expert_data_filename}.pkl")
        print("Possible filenames:")
        print("  - simple_dynamic_expert_final_1000.pkl")
        print("  - simple_dynamic_expert_final_500.pkl")
        print("  - Or other filename specified during data collection")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()