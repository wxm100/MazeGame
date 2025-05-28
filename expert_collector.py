# dynamic_expert_data_collector.py
"""
Dynamic expert data collection module
Uses dynamic expert agent to collect expert data for imitation learning
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
import pickle
import os
from tqdm import tqdm

# Import dynamic expert module
from dynamic_expert_agent import FixedDynamicBSPMiniGridEnv, SimpleDynamicExpert


class SimpleDynamicExpertDataCollector:
    """
    Simple dynamic expert data collector
    
    Responsible for collecting expert data using dynamic expert agent, supports:
    - Large-scale trajectory collection
    - Success rate monitoring
    - Failure reason analysis
    - Data quality control
    """
    
    def __init__(self, env_config: dict, data_dir: str = "./expert_data"):
        """
        Initialize data collector
        
        Args:
            env_config: Environment configuration dictionary
            data_dir: Data save directory
        """
        self.env_config = env_config.copy()
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Ensure obstacles are used
        self.env_config['barrier_count'] = max(1, self.env_config.get('barrier_count', 1))
        
        print(f"Initialized dynamic expert data collector")
        print(f"Environment config: {self.env_config}")
        print(f"Data directory: {self.data_dir}")
    
    def create_env(self, render_mode=None):
        """
        Create environment instance
        
        Args:
            render_mode: Render mode, None means no rendering
            
        Returns:
            Wrapped environment instance
        """
        env = FixedDynamicBSPMiniGridEnv(
            render_mode=render_mode,
            **self.env_config
        )
        return ImgObsWrapper(env)
    
    def collect_single_trajectory(self, env, max_steps: int = 500) -> Tuple[List[np.ndarray], List[int], bool, str]:
        """
        Collect single trajectory
        
        Args:
            env: Environment instance
            max_steps: Maximum steps per episode
            
        Returns:
            (observations, actions, success, failure_reason):
            - observations: Observation sequence
            - actions: Action sequence
            - success: Whether successful
            - failure_reason: Failure reason
        """
        observations = []
        actions = []
        
        # Reset environment and create expert
        obs, _ = env.reset()
        expert = SimpleDynamicExpert(env)
        
        for step in range(max_steps):
            # Record observation
            observations.append(obs.copy())
            
            # Get expert action
            action, path_found, reason = expert.get_next_action()
            
            if not path_found:
                return observations, actions, False, reason
            
            # Record action and execute
            actions.append(action)
            obs, reward, done, truncated, info = env.step(action)
            
            if done and reward > 0:  # Successfully reached goal
                return observations, actions, True, "success"
            elif done or truncated:
                return observations, actions, False, "collision_or_timeout"
        
        return observations, actions, False, "max_steps_exceeded"
    
    def collect_expert_data(self, 
                          num_trajectories: int, 
                          max_steps_per_episode: int = 600,
                          filename_prefix: str = "dynamic_expert_data",
                          success_rate_threshold: float = 0.3,
                          max_attempts_multiplier: int = 5) -> Dict[str, Any]:
        """
        Collect dynamic expert data
        
        Args:
            num_trajectories: Target number of successful trajectories
            max_steps_per_episode: Maximum steps per episode
            filename_prefix: Save filename prefix
            success_rate_threshold: Minimum success rate threshold, warning if below
            max_attempts_multiplier: Maximum attempts multiplier
            
        Returns:
            Collection statistics dictionary
        """
        print("Starting dynamic expert data collection")
        print(f"Target trajectories: {num_trajectories}")
        print(f"Max steps per episode: {max_steps_per_episode}")
        print(f"Success rate threshold: {success_rate_threshold:.1%}")
        
        # Initialize data storage
        all_observations = []
        all_actions = []
        successful_trajectories = 0
        failed_trajectories = 0
        failure_reasons = {}
        
        # Create environment
        env = self.create_env(render_mode=None)
        
        # Progress bar
        pbar = tqdm(total=num_trajectories, desc="Collecting trajectories", unit="traj")
        
        attempt_count = 0
        max_attempts = num_trajectories * max_attempts_multiplier
        
        while successful_trajectories < num_trajectories and attempt_count < max_attempts:
            attempt_count += 1
            
            # Collect single trajectory
            observations, actions, success, reason = self.collect_single_trajectory(
                env, max_steps_per_episode
            )
            
            if success and len(actions) > 0:
                # Successful trajectory: save data
                all_observations.extend(observations[:len(actions)])  # Ensure length match
                all_actions.extend(actions)
                successful_trajectories += 1
                pbar.update(1)
            else:
                # Failed trajectory: record reason
                failed_trajectories += 1
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            # Periodic statistical reporting
            if attempt_count % 100 == 0:
                current_success_rate = successful_trajectories / attempt_count
                pbar.set_postfix({
                    'Success rate': f'{current_success_rate:.2%}',
                    'Attempts': attempt_count
                })
                
                if current_success_rate < success_rate_threshold:
                    main_failure = max(failure_reasons, key=failure_reasons.get) if failure_reasons else "Unknown"
                    print(f"Warning: Success rate {current_success_rate:.2%} below threshold {success_rate_threshold:.2%}")
                    print(f"   Main failure reason: {main_failure}")
        
        pbar.close()
        env.close()
        
        # Calculate final statistics
        total_attempts = successful_trajectories + failed_trajectories
        success_rate = successful_trajectories / total_attempts if total_attempts > 0 else 0
        avg_steps = len(all_actions) / successful_trajectories if successful_trajectories > 0 else 0
        
        # Print results
        print("Data collection completed!")
        print(f"Successful trajectories: {successful_trajectories}")
        print(f"Failed attempts: {failed_trajectories}")
        print(f"Total success rate: {success_rate:.2%}")
        print(f"Total observations: {len(all_observations)}")
        print(f"Total actions: {len(all_actions)}")
        print(f"Average trajectory length: {avg_steps:.1f} steps")
        
        if failure_reasons:
            print(f"Failure reason statistics:")
            for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
                percentage = count / failed_trajectories * 100
                print(f"  {reason}: {count} ({percentage:.1f}%)")
        
        # Save data
        final_filename = f"{filename_prefix}_final_{num_trajectories}"
        self._save_data(all_observations, all_actions, final_filename)
        
        return {
            'observations': all_observations,
            'actions': all_actions,
            'num_trajectories': successful_trajectories,
            'success_rate': success_rate,
            'total_steps': len(all_actions),
            'avg_steps_per_trajectory': avg_steps,
            'failure_reasons': failure_reasons,
            'total_attempts': total_attempts
        }
    
    def _save_data(self, observations: List[np.ndarray], actions: List[int], filename: str):
        """
        Save data to file
        
        Args:
            observations: Observation data list
            actions: Action data list
            filename: Filename (without extension)
        """
        # Prepare data to save
        data = {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'env_config': self.env_config,
            'metadata': {
                'num_samples': len(actions),
                'obs_shape': observations[0].shape if observations else None,
                'action_space_size': max(actions) + 1 if actions else 0,
                'is_dynamic': True,  # Mark as dynamic environment data
                'has_obstacles': self.env_config.get('barrier_count', 0) > 0,
                'expert_type': 'SimpleDynamicExpert',
                'collection_version': '1.0'
            }
        }
        
        # Save file
        filepath = os.path.join(self.data_dir, f"{filename}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Data saved to: {filepath}")
        print(f"File size: {os.path.getsize(filepath) / (1024*1024):.1f} MB")
    
    def load_data(self, filename: str) -> Dict[str, Any]:
        """
        Load data file
        
        Args:
            filename: Filename (without extension)
            
        Returns:
            Data dictionary
        """
        filepath = os.path.join(self.data_dir, f"{filename}.pkl")
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def inspect_data(self, filename: str):
        """
        Inspect data file contents
        
        Args:
            filename: Filename (without extension)
        """
        try:
            data = self.load_data(filename)
            
            print(f"Data file inspection: {filename}.pkl")
            print(f"Observations: {len(data['observations'])}")
            print(f"Actions: {len(data['actions'])}")
            print(f"Observation shape: {data['observations'].shape}")
            print(f"Action range: {data['actions'].min()} - {data['actions'].max()}")
            print(f"Environment config: {data['env_config']}")
            print(f"Metadata: {data['metadata']}")
            
        except FileNotFoundError:
            print(f"File not found: {filename}.pkl")
        except Exception as e:
            print(f"Error reading file: {e}")


def main():
    """Main function - command line interface"""
    print("Dynamic Expert Data Collection Tool")
    
    # Environment configuration
    env_config = {
        'world_size': (18, 18),
        'room_count': 4,
        'goal_count': 1,
        'barrier_count': 6,  # Dynamic obstacle count
        'lava_count': 3,
        'lava_length': 4,
        'min_room_size': 5
    }
    
    print("Choose operation:")
    print("1. Collect dynamic expert data")
    print("2. Inspect existing data file")
    print("3. Test expert algorithm")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        # Collect data
        collector = SimpleDynamicExpertDataCollector(env_config)
        
        # Ask for collection parameters
        try:
            num_trajectories = int(input("Target trajectory count (default 1000): ") or "1000")
            max_steps = int(input("Max steps per episode (default 600): ") or "600")
            prefix = input("Filename prefix (default 'simple_dynamic_expert'): ") or "simple_dynamic_expert"
            
            print(f"Starting collection of {num_trajectories} trajectories...")
            
            # Start collection
            stats = collector.collect_expert_data(
                num_trajectories=num_trajectories,
                max_steps_per_episode=max_steps,
                filename_prefix=prefix
            )
            
            print(f"Collection completed!")
            print(f"Final success rate: {stats['success_rate']:.2%}")
            print(f"Average trajectory length: {stats['avg_steps_per_trajectory']:.1f} steps")
            
        except ValueError:
            print("Invalid input, please enter numbers")
        except KeyboardInterrupt:
            print("User interrupted collection")
        except Exception as e:
            print(f"Error during collection: {e}")
    
    elif choice == "2":
        # Inspect data file
        collector = SimpleDynamicExpertDataCollector(env_config)
        
        filename = input("Enter filename to inspect (without .pkl extension): ").strip()
        if filename:
            collector.inspect_data(filename)
        else:
            print("Filename cannot be empty")
    
    elif choice == "3":
        # Test expert algorithm
        print("Starting expert algorithm test...")
        from dynamic_expert_agent import test_dynamic_expert
        test_dynamic_expert()
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()