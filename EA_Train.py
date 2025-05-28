import numpy as np
import random
import time
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from minigrid.wrappers import ImgObsWrapper
from minigrid.core.constants import DIR_TO_VEC
from PCGEnv import DynamicBSPMiniGridEnv

SIMPLE_ENV_CONFIG = {
    "world_size": (12, 12),
    "room_count": 2,
    "goal_count": 1,
    "barrier_count": 0,
    "lava_count": 0,
    "lava_length": 0,
    "min_room_size": 3
}


def force_agent_goal_separation(env):
    base_env = env.unwrapped

    goals = []
    for x in range(base_env.grid.width):
        for y in range(base_env.grid.height):
            cell = base_env.grid.get(x, y)
            if cell and cell.type == 'goal':
                goals.append((x, y))

    if not goals:
        return False

    doors = []
    for x in range(base_env.grid.width):
        for y in range(base_env.grid.height):
            cell = base_env.grid.get(x, y)
            if cell and cell.type == 'door':
                doors.append((x, y))

    if not doors:
        return False

    empty_positions = []
    for x in range(1, base_env.grid.width - 1):
        for y in range(1, base_env.grid.height - 1):
            cell = base_env.grid.get(x, y)
            if cell is None:
                empty_positions.append((x, y))

    if not empty_positions:
        return False

    goal_pos = goals[0]

    separated_positions = []
    for pos in empty_positions:
        if not can_reach_without_doors(base_env, pos, goal_pos):
            distance = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
            separated_positions.append((distance, pos))

    if separated_positions:
        separated_positions.sort(reverse=True)
        best_pos = separated_positions[0][1]
        base_env.agent_pos = best_pos
        return True
    else:
        position_distances = []
        for pos in empty_positions:
            distance = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
            position_distances.append((distance, pos))

        position_distances.sort(reverse=True)
        best_pos = position_distances[0][1]
        base_env.agent_pos = best_pos
        return False


def can_reach_without_doors(base_env, start, goal):
    visited = set()
    queue = [start]

    while queue:
        current = queue.pop(0)
        if current == goal:
            return True

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_pos = (current[0] + dx, current[1] + dy)

            if not (0 <= next_pos[0] < base_env.grid.width and
                    0 <= next_pos[1] < base_env.grid.height):
                continue

            if next_pos in visited:
                continue

            cell = base_env.grid.get(*next_pos)

            if cell and (cell.type == 'wall' or
                         (cell.type == 'door' and not cell.is_open)):
                continue

            queue.append(next_pos)

    return False


class UltimateGoalRewardCalculator:
    def __init__(self):
        self.reach_goal_reward = 5000.0
        self.approach_goal_reward = 500.0
        self.move_away_goal_penalty = -300.0
        self.goal_distance_bonus = 200.0
        self.discover_goal_reward = 300.0
        self.open_door_reward = 2000.0
        self.face_door_reward = 400.0
        self.approach_door_reward = 150.0
        self.new_room_reward = 200.0
        self.new_area_reward = 10.0
        self.movement_reward = 2.0
        self.repeat_door_penalty = -200.0
        self.invalid_action_penalty = -20.0
        self.stuck_penalty = -50.0

    def calculate_reward(self, env, action, old_pos, new_pos, base_reward, agent_memory, step_count):
        total_reward = base_reward
        reward_breakdown = {'base': base_reward}

        try:
            goal_rewards = self._calculate_goal_rewards(env, old_pos, new_pos, agent_memory)
            total_reward += sum(goal_rewards.values())
            reward_breakdown.update(goal_rewards)

            door_rewards = self._calculate_door_rewards(env, action, old_pos, new_pos, agent_memory)
            total_reward += sum(door_rewards.values())
            reward_breakdown.update(door_rewards)

            exploration_rewards = self._calculate_exploration_rewards(env, old_pos, new_pos, agent_memory)
            total_reward += sum(exploration_rewards.values())
            reward_breakdown.update(exploration_rewards)

            penalties = self._calculate_penalties(env, action, agent_memory)
            total_reward += sum(penalties.values())
            reward_breakdown.update(penalties)

            return total_reward, reward_breakdown

        except Exception as e:
            return base_reward, {'base': base_reward, 'error': 0}

    def _calculate_goal_rewards(self, env, old_pos, new_pos, agent_memory):
        rewards = {}

        try:
            base_env = env.unwrapped

            goals = []
            for x in range(base_env.grid.width):
                for y in range(base_env.grid.height):
                    cell = base_env.grid.get(x, y)
                    if cell and cell.type == 'goal':
                        goals.append((x, y))

            if goals:
                goal_pos = goals[0]

                if new_pos == goal_pos:
                    rewards['reach_goal'] = self.reach_goal_reward

                old_dist = abs(old_pos[0] - goal_pos[0]) + abs(old_pos[1] - goal_pos[1])
                new_dist = abs(new_pos[0] - goal_pos[0]) + abs(new_pos[1] - goal_pos[1])

                if agent_memory and agent_memory.get('goal_discovered', False):
                    if new_dist < old_dist:
                        rewards['approach_goal'] = self.approach_goal_reward * (old_dist - new_dist)
                    elif new_dist > old_dist:
                        rewards['move_away_goal'] = self.move_away_goal_penalty * (new_dist - old_dist)

                    if new_dist <= 5:
                        rewards['goal_distance_bonus'] = self.goal_distance_bonus * (6 - new_dist) / 6

                if agent_memory and not agent_memory.get('goal_discovered', False):
                    if new_dist <= 4:
                        rewards['discover_goal'] = self.discover_goal_reward
                        agent_memory['goal_discovered'] = True

        except:
            pass

        return rewards

    def _calculate_door_rewards(self, env, action, old_pos, new_pos, agent_memory):
        rewards = {}

        try:
            base_env = env.unwrapped
            agent_pos = base_env.agent_pos
            agent_dir = base_env.agent_dir

            dir_vec = DIR_TO_VEC[agent_dir]
            front_pos = (agent_pos[0] + dir_vec[0], agent_pos[1] + dir_vec[1])

            if (0 <= front_pos[0] < base_env.grid.width and
                    0 <= front_pos[1] < base_env.grid.height):

                front_cell = base_env.grid.get(*front_pos)
                if front_cell and front_cell.type == 'door':

                    if not front_cell.is_open:
                        rewards['face_door'] = self.face_door_reward

                    if action == base_env.actions.toggle:
                        opened_doors = agent_memory.get('opened_doors', set()) if agent_memory else set()

                        if not front_cell.is_open and not front_cell.is_locked:
                            if front_pos not in opened_doors:
                                rewards['open_door'] = self.open_door_reward
                                if agent_memory:
                                    if 'opened_doors' not in agent_memory:
                                        agent_memory['opened_doors'] = set()
                                    agent_memory['opened_doors'].add(front_pos)

            door_approach = self._calculate_door_approach(env, old_pos, new_pos, agent_memory)
            if door_approach > 0:
                rewards['approach_door'] = door_approach

        except:
            pass

        return rewards

    def _calculate_door_approach(self, env, old_pos, new_pos, agent_memory):
        try:
            base_env = env.unwrapped

            doors = []
            opened_doors = agent_memory.get('opened_doors', set()) if agent_memory else set()

            for x in range(base_env.grid.width):
                for y in range(base_env.grid.height):
                    cell = base_env.grid.get(x, y)
                    if (cell and cell.type == 'door' and not cell.is_open
                            and (x, y) not in opened_doors):
                        doors.append((x, y))

            if doors:
                def min_door_distance(pos):
                    return min(abs(pos[0] - dx) + abs(pos[1] - dy) for dx, dy in doors)

                old_dist = min_door_distance(old_pos)
                new_dist = min_door_distance(new_pos)

                if new_dist < old_dist:
                    return self.approach_door_reward * (old_dist - new_dist)

            return 0
        except:
            return 0

    def _calculate_exploration_rewards(self, env, old_pos, new_pos, agent_memory):
        rewards = {}

        try:
            if agent_memory:
                visited = agent_memory.get('visited_positions', set())
                if new_pos not in visited:
                    rewards['new_area'] = self.new_area_reward
                    visited.add(new_pos)
                    agent_memory['visited_positions'] = visited

                    room_id = (new_pos[0] // 4, new_pos[1] // 4)
                    visited_rooms = agent_memory.get('visited_rooms', set())
                    if room_id not in visited_rooms:
                        visited_rooms.add(room_id)
                        agent_memory['visited_rooms'] = visited_rooms
                        rewards['new_room'] = self.new_room_reward

            if old_pos != new_pos:
                rewards['movement'] = self.movement_reward

        except:
            pass

        return rewards

    def _calculate_penalties(self, env, action, agent_memory):
        penalties = {}

        try:
            base_env = env.unwrapped

            if action == base_env.actions.toggle:
                agent_pos = base_env.agent_pos
                agent_dir = base_env.agent_dir
                dir_vec = DIR_TO_VEC[agent_dir]
                front_pos = (agent_pos[0] + dir_vec[0], agent_pos[1] + dir_vec[1])

                if agent_memory:
                    opened_doors = agent_memory.get('opened_doors', set())
                    if front_pos in opened_doors:
                        penalties['repeat_door'] = self.repeat_door_penalty

                if (0 <= front_pos[0] < base_env.grid.width and
                        0 <= front_pos[1] < base_env.grid.height):
                    front_cell = base_env.grid.get(*front_pos)
                    if not front_cell or front_cell.type != 'door':
                        penalties['invalid_action'] = self.invalid_action_penalty

            if agent_memory:
                recent_positions = agent_memory.get('recent_positions', deque())
                if len(recent_positions) >= 6:
                    unique_recent = len(set(recent_positions))
                    if unique_recent <= 2:
                        penalties['stuck'] = self.stuck_penalty

        except:
            pass

        return penalties


class UltimateGoalAgent:
    def __init__(self, neural_net):
        self.neural_net = neural_net
        self.reset_memory()
        self.door_forcing_probability = 0.8
        self.post_door_forward_steps = 3

    def reset_memory(self):
        self.memory = {
            'opened_doors': set(),
            'visited_positions': set(),
            'visited_rooms': set(),
            'recent_positions': deque(maxlen=8),
            'step_count': 0,
            'door_attempts': defaultdict(int),
            'goal_discovered': False,
            'just_opened_door': False,
            'post_door_steps': 0,
            'last_opened_door_pos': None
        }

    def get_action(self, env, obs):
        current_pos = env.unwrapped.agent_pos
        self.memory['recent_positions'].append(current_pos)
        self.memory['step_count'] += 1

        base_env = env.unwrapped

        if self.memory['just_opened_door'] and self.memory['post_door_steps'] < self.post_door_forward_steps:
            self.memory['post_door_steps'] += 1

            if self.memory['post_door_steps'] >= self.post_door_forward_steps:
                self.memory['just_opened_door'] = False
                self.memory['post_door_steps'] = 0
                self.memory['goal_discovered'] = True

            return base_env.actions.forward

        goal_pos = self._find_goal_position(env)
        if goal_pos:
            goal_distance = abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
            if goal_distance <= 5:
                self.memory['goal_discovered'] = True

        door_info = self._check_front_door(env)
        has_door, is_open, is_locked, door_pos = door_info

        if has_door and not is_open and not is_locked:
            attempts = self.memory['door_attempts'][door_pos]

            if attempts < 3 and np.random.random() < self.door_forcing_probability:
                self.memory['door_attempts'][door_pos] += 1
                self._prepare_post_door_behavior(door_pos)
                return base_env.actions.toggle

        nearest_door = self._find_nearest_unopened_door(env)
        if nearest_door and not self.memory.get('goal_discovered', False):
            move_action = self._move_towards_position(env, nearest_door)
            if move_action is not None:
                return move_action

        if np.random.random() < 0.1:
            actions = [base_env.actions.left, base_env.actions.right, base_env.actions.forward]
            return np.random.choice(actions)

        return self.neural_net.get_action(obs, env)

    def _prepare_post_door_behavior(self, door_pos):
        self.memory['just_opened_door'] = True
        self.memory['post_door_steps'] = 0
        self.memory['last_opened_door_pos'] = door_pos

    def verify_door_opened(self, env, door_pos):
        try:
            base_env = env.unwrapped
            if door_pos:
                cell = base_env.grid.get(*door_pos)
                if cell and cell.type == 'door' and cell.is_open:
                    self.memory['opened_doors'].add(door_pos)
                    return True
                else:
                    self.memory['just_opened_door'] = False
                    self.memory['post_door_steps'] = 0
                    return False
            return False
        except:
            return False

    def _check_front_door(self, env):
        try:
            base_env = env.unwrapped
            agent_pos = base_env.agent_pos
            agent_dir = base_env.agent_dir

            dir_vec = DIR_TO_VEC[agent_dir]
            front_pos = (agent_pos[0] + dir_vec[0], agent_pos[1] + dir_vec[1])

            if (0 <= front_pos[0] < base_env.grid.width and
                    0 <= front_pos[1] < base_env.grid.height):

                front_cell = base_env.grid.get(*front_pos)
                if front_cell and front_cell.type == 'door':
                    return True, front_cell.is_open, front_cell.is_locked, front_pos

            return False, False, False, None
        except:
            return False, False, False, None

    def _find_nearest_unopened_door(self, env):
        try:
            base_env = env.unwrapped
            agent_pos = base_env.agent_pos

            doors = []
            for x in range(base_env.grid.width):
                for y in range(base_env.grid.height):
                    cell = base_env.grid.get(x, y)
                    if (cell and cell.type == 'door' and not cell.is_open
                            and not cell.is_locked and (x, y) not in self.memory['opened_doors']):
                        distance = abs(x - agent_pos[0]) + abs(y - agent_pos[1])
                        doors.append((distance, (x, y)))

            if doors:
                doors.sort()
                return doors[0][1]
            return None
        except:
            return None

    def _find_goal_position(self, env):
        try:
            base_env = env.unwrapped

            for x in range(base_env.grid.width):
                for y in range(base_env.grid.height):
                    cell = base_env.grid.get(x, y)
                    if cell and cell.type == 'goal':
                        return (x, y)
            return None
        except:
            return None

    def _move_towards_position(self, env, target_pos):
        try:
            base_env = env.unwrapped
            agent_pos = base_env.agent_pos
            agent_dir = base_env.agent_dir

            dx = target_pos[0] - agent_pos[0]
            dy = target_pos[1] - agent_pos[1]

            if abs(dx) > abs(dy):
                target_dir = (1, 0) if dx > 0 else (-1, 0)
            else:
                target_dir = (0, 1) if dy > 0 else (0, -1)

            current_dir = DIR_TO_VEC[agent_dir]

            if np.array_equal(current_dir, target_dir):
                return base_env.actions.forward
            else:
                return base_env.actions.left
        except:
            return None


class SmartNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.random.randn(hidden_size) * 0.1
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.random.randn(output_size) * 0.1

    def preprocess_observation(self, obs, env=None):
        if len(obs.shape) == 3:
            flat_obs = obs.flatten()
        else:
            flat_obs = obs.flatten()

        goal_features = []
        if env:
            try:
                base_env = env.unwrapped
                agent_pos = base_env.agent_pos

                goal_pos = None
                for x in range(base_env.grid.width):
                    for y in range(base_env.grid.height):
                        cell = base_env.grid.get(x, y)
                        if cell and cell.type == 'goal':
                            goal_pos = (x, y)
                            break

                if goal_pos:
                    dx = (goal_pos[0] - agent_pos[0]) / base_env.grid.width
                    dy = (goal_pos[1] - agent_pos[1]) / base_env.grid.height
                    distance = abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])
                    normalized_distance = distance / (base_env.grid.width + base_env.grid.height)
                    goal_features = [dx, dy, normalized_distance, 1.0]
                else:
                    goal_features = [0.0, 0.0, 1.0, 0.0]
            except:
                goal_features = [0.0, 0.0, 1.0, 0.0]
        else:
            goal_features = [0.0, 0.0, 1.0, 0.0]

        combined_features = np.concatenate([flat_obs, goal_features])
        return combined_features

    def forward(self, x, env=None):
        processed_x = self.preprocess_observation(x, env)

        if len(processed_x.shape) > 1:
            processed_x = processed_x.flatten()

        if len(processed_x) < self.input_size:
            processed_x = np.pad(processed_x, (0, self.input_size - len(processed_x)))
        elif len(processed_x) > self.input_size:
            processed_x = processed_x[:self.input_size]

        h = np.tanh(np.dot(processed_x, self.w1) + self.b1)
        y = np.dot(h, self.w2) + self.b2

        exp_y = np.exp(y - np.max(y))
        probs = exp_y / np.sum(exp_y)
        return probs

    def get_action(self, x, env=None):
        probs = self.forward(x, env)
        return np.argmax(probs)

    def get_genome(self):
        return np.concatenate([
            self.w1.flatten(), self.b1,
            self.w2.flatten(), self.b2
        ])

    def set_genome(self, genome):
        idx = 0
        w1_size = self.input_size * self.hidden_size
        self.w1 = genome[idx:idx + w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size

        b1_size = self.hidden_size
        self.b1 = genome[idx:idx + b1_size]
        idx += b1_size

        w2_size = self.hidden_size * self.output_size
        self.w2 = genome[idx:idx + w2_size].reshape(self.hidden_size, self.output_size)
        idx += w2_size

        self.b2 = genome[idx:]


def ultimate_evaluate_individual(neural_net, env_config=SIMPLE_ENV_CONFIG,
                                 n_episodes=5, max_steps=150):
    reward_calculator = UltimateGoalRewardCalculator()

    total_fitness = 0
    total_rewards = 0
    goal_reached_count = 0
    doors_opened_count = 0
    success_episodes = 0
    separation_failures = 0

    for episode in range(n_episodes):
        env = DynamicBSPMiniGridEnv(**env_config)
        env = ImgObsWrapper(env)
        obs, _ = env.reset()

        separated = force_agent_goal_separation(env)
        if not separated:
            separation_failures += 1

        agent = UltimateGoalAgent(neural_net)

        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < max_steps:
            old_pos = env.unwrapped.agent_pos

            action = agent.get_action(env, obs)

            obs, base_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            new_pos = env.unwrapped.agent_pos

            total_reward, reward_breakdown = reward_calculator.calculate_reward(
                env, action, old_pos, new_pos, base_reward, agent.memory, steps
            )

            episode_reward += total_reward
            steps += 1

        total_rewards += episode_reward

        current_cell = env.unwrapped.grid.get(*new_pos)
        if current_cell and current_cell.type == 'goal':
            goal_reached_count += 1

        doors_opened_count += len(agent.memory.get('opened_doors', set()))

        if episode_reward > 1000:
            success_episodes += 1

        env.close()

    avg_reward = total_rewards / n_episodes
    goal_reached_rate = goal_reached_count / n_episodes
    avg_doors_opened = doors_opened_count / n_episodes
    success_rate = success_episodes / n_episodes

    fitness = (
            avg_reward * 0.2 +
            goal_reached_rate * 2000.0 +
            avg_doors_opened * 1000.0 +
            success_rate * 500.0
    )

    if separation_failures > 0:
        fitness *= 0.8

    metrics = {
        'avg_reward': avg_reward,
        'goal_reached_rate': goal_reached_rate,
        'avg_doors_opened': avg_doors_opened,
        'success_rate': success_rate,
        'fitness': fitness,
        'separation_failures': separation_failures
    }

    return fitness, metrics


def train_ultimate_goal_ea(env_config=SIMPLE_ENV_CONFIG,
                           population_size=60,
                           generations=50,
                           elite_size=6,
                           mutation_rate=0.15,
                           mutation_scale=0.3,
                           hidden_size=64):
    env = DynamicBSPMiniGridEnv(**env_config)
    env = ImgObsWrapper(env)
    input_size = np.prod(env.observation_space.shape) + 4
    output_size = env.action_space.n
    env.close()

    population = []
    for _ in range(population_size):
        nn = SmartNN(input_size, hidden_size, output_size)
        population.append(nn)

    fitness_scores = np.zeros(population_size)
    best_fitness_history = []
    avg_fitness_history = []
    goal_reached_history = []
    doors_opened_history = []
    success_rate_history = []

    best_genome = None
    best_fitness = -float('inf')

    start_time = time.time()

    for gen in range(generations):
        gen_start = time.time()

        goal_rates = []
        door_counts = []
        success_rates = []
        separation_failures = 0

        for i in range(population_size):
            fitness, metrics = ultimate_evaluate_individual(population[i], env_config)
            fitness_scores[i] = fitness
            goal_rates.append(metrics['goal_reached_rate'])
            door_counts.append(metrics['avg_doors_opened'])
            success_rates.append(metrics['success_rate'])
            separation_failures += metrics['separation_failures']

        max_fitness = np.max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        avg_goal_rate = np.mean(goal_rates)
        avg_doors = np.mean(door_counts)
        avg_success = np.mean(success_rates)

        best_fitness_history.append(max_fitness)
        avg_fitness_history.append(avg_fitness)
        goal_reached_history.append(avg_goal_rate)
        doors_opened_history.append(avg_doors)
        success_rate_history.append(avg_success)

        best_idx = np.argmax(fitness_scores)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_genome = population[best_idx].get_genome().copy()

        if gen < generations - 1:
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            elites = [population[i] for i in elite_indices]

            min_fitness = min(0, np.min(fitness_scores))
            adjusted_fitness = fitness_scores - min_fitness + 1e-6
            selection_probs = adjusted_fitness / np.sum(adjusted_fitness)

            new_population = elites.copy()

            while len(new_population) < population_size:
                parent_indices = np.random.choice(
                    population_size, size=2, p=selection_probs, replace=False
                )
                parent1 = population[parent_indices[0]]
                parent2 = population[parent_indices[1]]

                genome1 = parent1.get_genome()
                genome2 = parent2.get_genome()

                crossover_point = np.random.randint(1, len(genome1))
                child_genome = np.concatenate([genome1[:crossover_point],
                                               genome2[crossover_point:]])

                mutation_mask = np.random.random(size=child_genome.shape) < mutation_rate
                mutations = np.random.normal(scale=mutation_scale, size=child_genome.shape)
                child_genome = child_genome + mutation_mask * mutations

                child = SmartNN(input_size, hidden_size, output_size)
                child.set_genome(child_genome)
                new_population.append(child)

            population = new_population

    best_model = SmartNN(input_size, hidden_size, output_size)
    best_model.set_genome(best_genome)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 2)
    plt.plot(goal_reached_history, 'g-', linewidth=3)
    plt.xlabel('Generation')
    plt.ylabel('Goal Reach Rate')
    plt.title('Goal Reach Rate')
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(doors_opened_history, 'orange', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Average Doors Opened')
    plt.title('Door Opening Performance')
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(success_rate_history, 'purple', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Success Rate')
    plt.title('Overall Success Rate')
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(goal_reached_history, 'g-', linewidth=2, label='Goal Rate')
    plt.plot(doors_opened_history, 'orange', linewidth=2, label='Doors/2')
    plt.plot(success_rate_history, 'purple', linewidth=2, label='Success')
    plt.xlabel('Generation')
    plt.ylabel('Performance Metrics')
    plt.title('Key Metrics Comparison')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 6)
    normalized_fitness = np.array(best_fitness_history) / max(best_fitness_history)
    plt.plot(normalized_fitness, 'r-', linewidth=2, label='Fitness (norm)')
    plt.plot(goal_reached_history, 'g-', linewidth=2, label='Goal Rate')
    plt.xlabel('Generation')
    plt.ylabel('Normalized Score')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("ultimate_goal_training.png", dpi=300, bbox_inches='tight')
    plt.close()

    return best_model


def visualize_ultimate_agent(agent, env_config=SIMPLE_ENV_CONFIG, episodes=3, delay=0.3):
    from PIL import Image

    total_rewards = []
    goal_reached_list = []
    doors_opened_list = []
    separation_success = []
    frames = []

    reward_calculator = UltimateGoalRewardCalculator()

    for ep in range(episodes):
        env = DynamicBSPMiniGridEnv(render_mode="human", **env_config)
        env = ImgObsWrapper(env)
        obs, _ = env.reset()

        separated = force_agent_goal_separation(env)
        separation_success.append(separated)

        ultimate_agent = UltimateGoalAgent(agent)

        done = False
        episode_reward = 0
        steps = 0
        max_steps = 200

        goal_reached = False
        doors_opened_count = 0

        while not done and steps < max_steps:
            old_pos = env.unwrapped.agent_pos

            action = ultimate_agent.get_action(env, obs)

            obs, base_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            new_pos = env.unwrapped.agent_pos

            if action == env.unwrapped.actions.toggle and ultimate_agent.memory.get('last_opened_door_pos'):
                door_pos = ultimate_agent.memory['last_opened_door_pos']
                if ultimate_agent.verify_door_opened(env, door_pos):
                    doors_opened_count += 1

            env.render()

            env.unwrapped.render_mode = "rgb_array"
            frame = env.render()
            frames.append(Image.fromarray(frame))
            env.unwrapped.render_mode = "human"

            total_reward, reward_breakdown = reward_calculator.calculate_reward(
                env, action, old_pos, new_pos, base_reward, ultimate_agent.memory, steps
            )

            episode_reward += total_reward

            if 'reach_goal' in reward_breakdown and reward_breakdown['reach_goal'] > 0:
                goal_reached = True

            steps += 1
            time.sleep(delay)

        total_rewards.append(episode_reward)
        goal_reached_list.append(goal_reached)
        doors_opened_list.append(doors_opened_count)

        env.close()

    if frames:
        frames[0].save("agent_demo.gif", save_all=True, append_images=frames[1:], duration=int(delay * 1000), loop=0)


def main():
    np.random.seed(42)
    random.seed(42)

    mode = input("Select mode (1-5): ").strip()

    if mode == "1":
        generations = int(input("Training generations (default 50): ") or "50")

        best_agent = train_ultimate_goal_ea(
            env_config=SIMPLE_ENV_CONFIG,
            population_size=60,
            generations=generations,
            elite_size=6,
            mutation_rate=0.15,
            mutation_scale=0.3,
            hidden_size=64
        )

        best_genome = best_agent.get_genome()
        np.save("ultimate_goal_genome.npy", best_genome)

        model_config = {
            "input_size": best_agent.input_size,
            "hidden_size": best_agent.hidden_size,
            "output_size": best_agent.output_size
        }
        np.save("ultimate_goal_config.npy", model_config)

        visualize_ultimate_agent(best_agent, episodes=3)

    elif mode == "2":
        try:
            model_config = np.load("ultimate_goal_config.npy", allow_pickle=True).item()
            best_genome = np.load("ultimate_goal_genome.npy")

            agent = SmartNN(
                model_config["input_size"],
                model_config["hidden_size"],
                model_config["output_size"]
            )
            agent.set_genome(best_genome)

            episodes = int(input("Demo episodes (default 3): ") or "3")
            delay = float(input("Action delay (default 0.3): ") or "0.3")

            visualize_ultimate_agent(agent, episodes=episodes, delay=delay)

        except FileNotFoundError:
            print("Model files not found")

    elif mode == "3":
        best_agent = train_ultimate_goal_ea(
            env_config=SIMPLE_ENV_CONFIG,
            population_size=30,
            generations=25,
            elite_size=4,
            mutation_rate=0.2,
            mutation_scale=0.4,
            hidden_size=48
        )

        visualize_ultimate_agent(best_agent, episodes=2, delay=0.5)

    elif mode == "4":
        env = DynamicBSPMiniGridEnv(**SIMPLE_ENV_CONFIG)
        env = ImgObsWrapper(env)
        input_size = np.prod(env.observation_space.shape) + 4
        output_size = env.action_space.n
        env.close()

        neural_net = SmartNN(input_size, 32, output_size)

        for i in range(5):
            fitness, metrics = ultimate_evaluate_individual(neural_net, n_episodes=2)

    elif mode == "5":
        try:
            model_config = np.load("ultimate_goal_config.npy", allow_pickle=True).item()
            best_genome = np.load("ultimate_goal_genome.npy")

            agent = SmartNN(
                model_config["input_size"],
                model_config["hidden_size"],
                model_config["output_size"]
            )
            agent.set_genome(best_genome)

            filename = input("GIF filename (default agent.gif): ") or "agent.gif"
            max_steps = int(input("Max steps (default 150): ") or "150")

            print("Recording GIF...")
            record_agent_simple_gif(agent, filename=filename, max_steps=max_steps)

        except FileNotFoundError:
            print("Model files not found")


def record_agent_simple_gif(agent, env_config=SIMPLE_ENV_CONFIG, filename="agent.gif", max_steps=150):
    from PIL import Image

    env = DynamicBSPMiniGridEnv(render_mode="rgb_array", **env_config)
    env = ImgObsWrapper(env)
    obs, _ = env.reset()

    force_agent_goal_separation(env)
    ultimate_agent = UltimateGoalAgent(agent)

    frames = []
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = ultimate_agent.get_action(env, obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        frame = env.render()
        frames.append(Image.fromarray(frame))
        steps += 1

    env.close()

    if frames:
        frames[0].save(filename, save_all=True, append_images=frames[1:], duration=500, loop=0)
        print(f"Saved {filename} with {len(frames)} frames")


if __name__ == "__main__":
    main()