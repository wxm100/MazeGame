# dynamic_expert_agent.py
"""
Dynamic expert agent module
Contains dynamic environment fixes and expert decision algorithms
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import heapq
from minigrid.core.world_object import Goal, Lava, Wall, Door, Ball
from PCGEnv import DynamicBSPMiniGridEnv
from Heuristicagent import astar, manhattan


class FixedDynamicBSPMiniGridEnv(DynamicBSPMiniGridEnv):
    """
    Fixed dynamic environment wrapper
    Fixes the issue where step0 function was not called in original PCGEnv
    """
    
    def step(self, action: int):
        """Override step function to ensure ball movement logic is executed"""
        # Manually execute ball movement logic (copied from original step0 function)
        for b in self.obstacles:
            ox, oy = b.cur_pos
            size_x, size_y = 3, 3
            top_x = min(max(ox-1, 1), self.width - size_x - 1)
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
                # fallback: put the ball back where it was (clamped)
                ox = min(max(ox, 1), self.width-2)
                oy = min(max(oy, 1), self.height-2)
                b.cur_pos = (ox, oy)
                self.grid.set(ox, oy, b)

        # Call parent step function to handle agent movement
        obs, reward, term, trunc, info = super().step(action)
        
        # Check collision
        if action == self.actions.forward:
            front = self.grid.get(*self.front_pos)
            if front and front.type == "ball":
                term = True

        return obs, reward, term, trunc, info


class SimpleDynamicExpert:
    """
    Simple dynamic expert agent
    
    Uses progressive safety strategy for dynamic obstacle avoidance:
    1. Cautious mode: avoid areas 3+ cells around balls
    2. Normal mode: avoid adjacent positions to balls
    3. Aggressive mode: only avoid overlapping with balls
    4. Direct mode: ignore balls, navigate directly
    """
    
    def __init__(self, env):
        """
        Initialize dynamic expert
        
        Args:
            env: Environment instance, should be FixedDynamicBSPMiniGridEnv type
        """
        self.env = env
        self.safety_mode = "normal"
        
    def get_ball_danger_level(self, pos: Tuple[int, int]) -> int:
        """
        Evaluate danger level of a position
        
        Args:
            pos: Position coordinates (x, y) to evaluate
            
        Returns:
            Danger level:
            - 0: Safe (distance 3+ cells)
            - 1: Caution (distance 2 cells)
            - 2: Dangerous (distance 1 cell)
            - 3: Extremely dangerous (distance 0 cells, overlap)
        """
        min_distance = float('inf')
        for ball in self.env.unwrapped.obstacles:
            dist = manhattan(pos, ball.cur_pos)
            min_distance = min(min_distance, dist)
        
        if min_distance == 0:
            return 3  # Extremely dangerous: overlapping with ball
        elif min_distance == 1:
            return 2  # Dangerous: adjacent
        elif min_distance == 2:
            return 1  # Caution: 2 cells distance
        else:
            return 0  # Safe: 3+ cells
    
    def is_position_acceptable(self, pos: Tuple[int, int], safety_level: str = "normal") -> bool:
        """
        Check if position is acceptable based on safety level
        
        Args:
            pos: Position coordinates
            safety_level: Safety level ("cautious", "normal", "aggressive")
            
        Returns:
            bool: Whether position is acceptable
        """
        danger_level = self.get_ball_danger_level(pos)
        
        if safety_level == "aggressive":
            return danger_level < 3  # Only avoid overlap
        elif safety_level == "normal":
            return danger_level < 2  # Avoid adjacent
        else:  # cautious
            return danger_level == 0  # Only accept completely safe positions
    
    def find_goal_position(self) -> Optional[Tuple[int, int]]:
        """
        Find goal position
        
        Returns:
            Goal position coordinates, None if not found
        """
        grid = self.env.unwrapped.grid
        width, height = self.env.unwrapped.width, self.env.unwrapped.height
        
        for x in range(width):
            for y in range(height):
                if isinstance(grid.get(x, y), Goal):
                    return (x, y)
        return None
    
    def find_path_with_safety_level(self, start: Tuple[int, int], goal: Tuple[int, int], 
                                   safety_level: str) -> List[Tuple[int, int]]:
        """
        Find path using specified safety level
        
        Args:
            start: Start position
            goal: Goal position
            safety_level: Safety level
            
        Returns:
            Path list, empty list if no path found
        """
        grid = self.env.unwrapped.grid
        width, height = self.env.unwrapped.width, self.env.unwrapped.height
        
        def is_safe_neighbor(pos):
            """Check if neighbor position is safe"""
            x, y = pos
            if not (0 <= x < width and 0 <= y < height):
                return False
            
            # Check static obstacles
            cell = grid.get(x, y)
            if isinstance(cell, (Wall, Lava)):
                return False
            
            # Check dynamic safety
            return self.is_position_acceptable(pos, safety_level)
        
        # A* search
        g_score = {start: 0}
        f_score = {start: manhattan(start, goal)}
        open_set = [(f_score[start], start)]
        came_from = {}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            # Explore neighbors
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not is_safe_neighbor(neighbor):
                    continue
                
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + manhattan(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []
    
    def get_safe_path_with_flexibility(self) -> Tuple[List[Tuple[int, int]], str]:
        """
        Get path to goal using progressive safety strategy
        
        Returns:
            (path, safety_mode_used): Path and safety mode used
        """
        start = tuple(self.env.unwrapped.agent_pos)
        goal = self.find_goal_position()
        
        if not goal:
            return [], "no_goal"
        
        # Try different safety levels, from conservative to aggressive
        safety_levels = ["cautious", "normal", "aggressive"]
        
        for safety_mode in safety_levels:
            path = self.find_path_with_safety_level(start, goal, safety_mode)
            if path:
                return path, safety_mode
        
        # If all safety levels fail, try direct A* (ignore balls)
        grid = self.env.unwrapped.grid
        width, height = self.env.unwrapped.width, self.env.unwrapped.height
        path = astar(start, goal, grid, width, height)
        return path, "direct" if path else "impossible"
    
    def get_escape_action(self) -> Optional[int]:
        """
        Get action to escape danger
        
        Returns:
            Escape action, None if cannot escape
        """
        current_pos = tuple(self.env.unwrapped.agent_pos)
        agent_dir = self.env.unwrapped.agent_dir
        
        # Try 4 directions, find position with lowest danger level
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        best_pos = None
        min_danger_level = 10
        
        for dx, dy in directions:
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Check bounds and static obstacles
            if (0 <= new_pos[0] < self.env.unwrapped.width and 
                0 <= new_pos[1] < self.env.unwrapped.height):
                cell = self.env.unwrapped.grid.get(*new_pos)
                if not isinstance(cell, (Wall, Lava)):
                    
                    danger_level = self.get_ball_danger_level(new_pos)
                    
                    if danger_level < min_danger_level:
                        min_danger_level = danger_level
                        best_pos = new_pos
        
        if best_pos and min_danger_level < 3:  # As long as not extremely dangerous
            return self.calculate_move_action(current_pos, best_pos, agent_dir)
        return None
    
    def calculate_move_action(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int], 
                            agent_dir: int) -> int:
        """
        Calculate action needed to move from current to target position
        
        Args:
            current_pos: Current position
            target_pos: Target position
            agent_dir: Current agent direction
            
        Returns:
            Action ID
        """
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Calculate target direction
        target_dir_map = {
            (1, 0): 0,   # Right
            (0, 1): 1,   # Down
            (-1, 0): 2,  # Left
            (0, -1): 3   # Up
        }
        
        if (dx, dy) not in target_dir_map:
            return self.env.unwrapped.actions.forward  # Wait or invalid move
        
        target_dir = target_dir_map[(dx, dy)]
        
        # If direction is correct, check if need to open door or move forward
        if agent_dir == target_dir:
            cell = self.env.unwrapped.grid.get(*target_pos)
            if isinstance(cell, Door) and not cell.is_open:
                return self.env.unwrapped.actions.toggle
            else:
                return self.env.unwrapped.actions.forward
        else:
            # Need to turn
            diff = (target_dir - agent_dir) % 4
            if diff == 1:
                return self.env.unwrapped.actions.right
            elif diff == 3:
                return self.env.unwrapped.actions.left
            else:  # diff == 2 (180 degree turn)
                return self.env.unwrapped.actions.right
    
    def get_next_action(self) -> Tuple[int, bool, str]:
        """
        Get next action (main interface)
        
        Returns:
            (action, success, reason): Action ID, success flag, decision reason
        """
        current_pos = tuple(self.env.unwrapped.agent_pos)
        agent_dir = self.env.unwrapped.agent_dir
        
        # Evaluate current position danger
        current_danger = self.get_ball_danger_level(current_pos)
        
        # Strategy 1: If extremely dangerous (overlapping with ball), escape immediately
        if current_danger >= 3:
            escape_action = self.get_escape_action()
            if escape_action is not None:
                return escape_action, True, "emergency_escape"
            else:
                return 0, False, "trapped"
        
        # Strategy 2: Get flexible path planning
        path, safety_mode = self.get_safe_path_with_flexibility()
        
        if not path or len(path) < 2:
            return 0, False, "no_path_available"
        
        next_pos = path[1]
        
        # Strategy 3: Check immediate safety of next step
        next_danger = self.get_ball_danger_level(next_pos)
        
        # If next step is extremely dangerous, try waiting
        if next_danger >= 3:
            return self.env.unwrapped.actions.right, True, "waiting_turn"
        
        # Execute move
        action = self.calculate_move_action(current_pos, next_pos, agent_dir)
        return action, True, f"moving_{safety_mode}"


def test_dynamic_expert():
    """Simple test example for dynamic expert algorithm"""
    env_config = {
        'world_size': (12, 12),
        'room_count': 2,
        'goal_count': 1,
        'barrier_count': 2,
        'lava_count': 1,
        'lava_length': 3,
        'min_room_size': 5
    }
    
    # Create environment and expert
    env = FixedDynamicBSPMiniGridEnv(render_mode="human", **env_config)
    expert = SimpleDynamicExpert(env)
    
    obs, _ = env.reset()
    
    print("Testing dynamic expert algorithm...")
    print(f"Agent position: {env.unwrapped.agent_pos}")
    print(f"Ball positions: {[ball.cur_pos for ball in env.unwrapped.obstacles]}")
    
    # Test run
    for step in range(50):
        action, success, reason = expert.get_next_action()
        
        if not success:
            print(f"Expert failed: {reason}")
            break
        
        print(f"Step {step+1}: action={action}, reason={reason}")
        obs, reward, done, truncated, info = env.step(action)
        
        if done:
            if reward > 0:
                print(f"Success! Steps: {step+1}")
            else:
                print(f"Failed")
            break
        
        if truncated:
            print(f"Timeout")
            break
        
        import time
        time.sleep(0.5)
    
    env.close()


if __name__ == "__main__":
    # Run test
    test_dynamic_expert()