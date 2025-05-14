import heapq
from PCGEnv import DynamicBSPMiniGridEnv
from minigrid.core.world_object import Goal, Lava, Wall, Door

# Heuristic A* search to plan a safe path to the goal, avoiding lava, walls, and handling doors

def astar(start, goal, grid, width, height):
    g_score = {start: 0}
    f_score = {start: manhattan(start, goal)}

    open_set = [(f_score[start], start)]
    came_from = {}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in neighbors(current, grid, width, height):
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + manhattan(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(pos, grid, width, height):
    x, y = pos
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            cell = grid.get(nx, ny)
            # Allow movement into empty, open doors, closed doors (we'll open them at runtime), and goals
            if isinstance(cell, (Wall, Lava)):
                continue
            yield (nx, ny)


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]


def rotate_towards(agent_dir, target_dir):
    if target_dir is None:
        return []
    diff = (target_dir - agent_dir) % 4
    if diff == 3:
        return ['left']
    elif diff == 1:
        return ['right']
    elif diff == 2:
        return ['right', 'right']
    return []


def direction_from_to(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 1 and dy == 0:
        return 0
    if dx == 0 and dy == 1:
        return 1
    if dx == -1 and dy == 0:
        return 2
    if dx == 0 and dy == -1:
        return 3
    return None


def execute_step(env, current, target):
    dx, dy = target[0] - current[0], target[1] - current[1]
    # Break multi-tile straight moves into unit steps
    steps = []
    if dx == 0 and abs(dy) > 1:
        sign = 1 if dy > 0 else -1
        steps = [(current[0], current[1] + sign * i) for i in range(1, abs(dy) + 1)]
    elif dy == 0 and abs(dx) > 1:
        sign = 1 if dx > 0 else -1
        steps = [(current[0] + sign * i, current[1]) for i in range(1, abs(dx) + 1)]
    else:
        steps = [target]

    for step in steps:
        # Compute direction and rotate
        desired = direction_from_to(current, step)
        for act in rotate_towards(env.unwrapped.agent_dir, desired):
            env.step(getattr(env.actions, act))

        # If there's a closed door ahead, toggle it open first
        cell = env.unwrapped.grid.get(*step)
        if isinstance(cell, Door) and not cell.is_open:
            env.step(env.actions.toggle)

        # Move forward into the cell (or through opened door)
        env.step(env.actions.forward)
        current = tuple(env.unwrapped.agent_pos)

    return current


def move_along_path(env, path):
    current = tuple(env.unwrapped.agent_pos)
    for target in path[1:]:
        current = execute_step(env, current, target)


def main():
    env = DynamicBSPMiniGridEnv(
        world_size=(18,18),
        room_count=5,
        goal_count=1,
        barrier_count=5,
        lava_count=2,
        lava_length=4,
        render_mode="human"
    )

    obs, _ = env.reset()
    grid = env.unwrapped.grid
    width, height = env.unwrapped.width, env.unwrapped.height
    start = tuple(env.unwrapped.agent_pos)

    goal = next(((x, y) for x in range(width) for y in range(height)
                 if isinstance(grid.get(x, y), Goal)), None)
    if goal is None:
        print("No goal found on the map.")
        return

    path = astar(start, goal, grid, width, height)
    if not path:
        print("No safe path to the goal found.")
    else:
        print(f"Path to goal found: {path}")
        move_along_path(env, path)

    env.close()

if __name__ == "__main__":
    main()
