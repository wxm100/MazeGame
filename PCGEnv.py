import random
from typing import Tuple, List, Optional

from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal, Door, Ball, Lava
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.manual_control import ManualControl
from minigrid.core.constants import COLOR_NAMES
import numpy as np
from gymnasium.spaces import Discrete

class DynamicBSPMiniGridEnv(MiniGridEnv):
    """
    MiniGrid env with iterative BSP splitting, dynamic obstacles, and lava. 
    a) no keys
    b) all doors are closed but not locked
git 
    Args:
      world_size: (width, height)
      room_count: target number of rooms
      goal_count: number of green goals
      barrier_count: number of moving Ball obstacles
      lava_count: number of Lava segments
      lava_length: length of each segment
      min_room_size: minimum dimension (width/height) for a room
    """
    """
    New fix (not yet fully fixed): 
        a) doors are only placed on unbroken wall segments, never at corridor crossings.
    """

    def __init__(
        self,
        world_size: Tuple[int, int] = (16, 16),
        room_count: int = 4,
        goal_count: int = 1,
        barrier_count: int = 1,
        lava_count: int = 1,
        lava_length: int = 5,
        min_room_size: int = 5,
        max_steps: Optional[int] = None,
        **kwargs
    ):
        w, h = world_size
        mission = (
            #f"collect all {goal_count} goals, avoid {barrier_count} obstacles, nav lava"
            ""
        )
        ms = MissionSpace(mission_func=lambda: mission)

        if max_steps is None:
            max_steps = 8 * w * h

        super().__init__(
            mission_space=ms,
            width=w,
            height=h,
            max_steps=max_steps,
            **kwargs
        )

        self.room_count    = room_count
        self.goal_count    = goal_count
        self.n_obstacles   = barrier_count
        self.lava_count    = lava_count
        self.lava_length   = lava_length
        self.min_room_size = min_room_size

        # actions: turn left/turn right/forward
        self.action_space = Discrete(self.actions.toggle + 1)
        self.obstacles: List[Ball] = []
    
    def _gen_grid(self, width: int, height: int):
        # 1) Outer walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # 2) Iterative BSP splitting with “door-safe” cuts only
        regions = [(0, 0, width, height)]
        splits = self.room_count - 1

        while splits > 0:
            safe_regions = []
            for idx, (x, y, w, h) in enumerate(regions):
                can_v = (w >= 2*self.min_room_size + 1 and h >= 3)
                can_h = (h >= 2*self.min_room_size + 1 and w >= 3)
                if not (can_v or can_h):
                    continue

                v_cuts, h_cuts = [], []

                # vertical safe cuts (unchanged) …
                if can_v:
                    for cut in range(self.min_room_size, w - self.min_room_size):
                        blocked = False
                        for j in range(y, y + h):
                            if isinstance(self.grid.get(x+cut, j), Door):
                                blocked = True; break
                            if isinstance(self.grid.get(x+cut-1, j), Door) \
                            or isinstance(self.grid.get(x+cut+1, j), Door):
                                blocked = True; break
                        if not blocked:
                            v_cuts.append(cut)

                # horizontal safe cuts (with **endpoint** checks)
                if can_h:
                    for cut in range(self.min_room_size, h - self.min_room_size):
                        blocked = False

                        # 1) scan the would-be wall line for overwrites/abutments
                        for i in range(x, x + w):
                            # never overwrite a door
                            if isinstance(self.grid.get(i, y+cut), Door):
                                blocked = True; break
                            # never abut a door above/below
                            if isinstance(self.grid.get(i, y+cut-1), Door) \
                            or isinstance(self.grid.get(i, y+cut+1), Door):
                                blocked = True; break
                        if blocked:
                            continue

                        # 2) **endpoint** checks: don't drown external corridors
                        #    left endpoint: (x, y+cut) neighbor at (x-1, y+cut)
                        if x-1 >= 0:
                            nbr = self.grid.get(x-1, y+cut)
                            if isinstance(nbr, Door):
                                continue

                        #    right endpoint
                        if x + w < width:
                            nbr = self.grid.get(x+w, y+cut)
                            if isinstance(nbr, Door):
                                continue

                        # passed all tests!
                        h_cuts.append(cut)

                if v_cuts or h_cuts:
                    safe_regions.append((idx, x, y, w, h, v_cuts, h_cuts))

            # stop if nothing safe remains
            if not safe_regions:
                break

            # pick largest
            idx, x, y, w, h, v_cuts, h_cuts = max(
                safe_regions, key=lambda t: t[3] * t[4]
            )
            regions.pop(idx)

            # split …
            if v_cuts and (not h_cuts or w >= h):
                cut = random.choice(v_cuts)
                door_rows = [
                    j for j in range(y+1, y+h-1)
                    if self.grid.get(x+cut-1, j) is None
                    and self.grid.get(x+cut+1, j) is None
                ]
                door_j = random.choice(door_rows) if door_rows else random.randint(y+1, y+h-2)
                for j in range(y, y+h):
                    self.grid.set(x+cut, j,
                                Door(random.choice(COLOR_NAMES)) if j==door_j else Wall())
                regions += [(x, y, cut, h), (x+cut+1, y, w-cut-1, h)]
            else:
                cut = random.choice(h_cuts)
                door_cols = [
                    i for i in range(x+1, x+w-1)
                    if self.grid.get(i, y+cut-1) is None
                    and self.grid.get(i, y+cut+1) is None
                ]
                door_i = random.choice(door_cols) if door_cols else random.randint(x+1, x+w-2)
                for i in range(x, x+w):
                    self.grid.set(i, y+cut,
                                Door(random.choice(COLOR_NAMES)) if i==door_i else Wall())
                regions += [(x, y, w, cut), (x, y+cut+1, w, h-cut-1)]

            splits -= 1

        # 3) Carve interior floors
        for rx, ry, rw, rh in regions:
            for i in range(rx+1, rx+rw-1):
                for j in range(ry+1, ry+rh-1):
                    self.grid.set(i, j, None)

        # 4) Lava segments
        lava_cands = []
        for rx, ry, rw, rh in regions:
            iw, ih = rw-2, rh-2
            if iw >= self.lava_length:
                for row in range(ry+1, ry+rh-1):
                    mstart = rx+1 + (iw - self.lava_length)
                    if mstart >= rx+1:
                        lava_cands.append(("h", rx, ry, rw, rh, row))
            if ih >= self.lava_length:
                for col in range(rx+1, rx+rw-1):
                    mstart = ry+1 + (ih - self.lava_length)
                    if mstart >= ry+1:
                        lava_cands.append(("v", rx, ry, rw, rh, col))

        random.shuffle(lava_cands)
        for _ in range(self.lava_count):
            if not lava_cands:
                break
            orient, rx, ry, rw, rh, fixed = lava_cands.pop()
            if orient == "h":
                row = fixed
                smin = rx+1
                smax = rx+1 + (rw-2-self.lava_length)
                start = random.randint(smin, smax)
                for i in range(start, start+self.lava_length):
                    self.grid.set(i, row, Lava())
            else:
                col = fixed
                smin = ry+1
                smax = ry+1 + (rh-2-self.lava_length)
                start = random.randint(smin, smax)
                for j in range(start, start+self.lava_length):
                    self.grid.set(col, j, Lava())

        # 5) Goals
        floor = [
            (i, j)
            for i in range(1, width-1)
            for j in range(1, height-1)
            if self.grid.get(i, j) is None
        ]
        random.shuffle(floor)
        for _ in range(min(self.goal_count, len(floor))):
            gx, gy = floor.pop()
            self.grid.set(gx, gy, Goal())

        # 6) Moving Balls
        self.obstacles = []
        random.shuffle(floor)
        for _ in range(min(self.n_obstacles, len(floor))):
            ox, oy = floor.pop()
            b = Ball()
            self.grid.set(ox, oy, b)
            b.cur_pos = (ox, oy)
            self.obstacles.append(b)

        # 7) Agent
        random.shuffle(floor)
        if floor:
            self.agent_pos = floor.pop()
        else:
            fx, fy, fw, fh = regions[0]
            self.agent_pos = (fx + fw//2, fy + fh//2)
        self.agent_dir = random.randint(0, 3)



    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # record initial distance to nearest goal
        self._goals = [(i,j) for i in range(self.width)
                            for j in range(self.height)
                            if isinstance(self.grid.get(i,j), Goal)]
        # record the history path
        self.visited = set()
        start = tuple(self.agent_pos)
        self.visited.add(start)

        self._old_dist = self._dist_to_goals(self.agent_pos)
        return obs, info
    
    # considering the scenario of multiple goals
    def _dist_to_goals(self, pos):
        # Manhattan distance to closest goal
        return min(abs(pos[0]-gx) + abs(pos[1]-gy) for gx,gy in self._goals)

    # using default step function
    def step(self, action: int):
        # move Balls
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
                # fallback: put the ball back where it was (clamped)
                ox = min(max(ox, 1), self.width-2)
                oy = min(max(oy, 1), self.height-2)
                b.cur_pos = (ox, oy)
                self.grid.set(ox, oy, b)

        obs, reward, term, trunc, info = super().step(action)
        if action == self.actions.forward:
            front = self.grid.get(*self.front_pos)
            if front and front.type=="ball":
                #reward = -1
                term = True

        #0) reset reward
        # reward =0

        # 1) Distance-based reward
        # old_dist = self._old_dist
        # new_dist = self._dist_to_goals(tuple(self.agent_pos))
        # # reward for getting closer, vise verse
        # reward += 0.1 * (old_dist - new_dist)
        # self._old_dist = new_dist
        # if new_dist == 0:
        #     reward+=10


        # # 2) Small step penalty to avoid wandering around aimlessly
        # reward -= 0.01  

        # # 3) Bonus for opening a door
        # if action == self.actions.toggle:
        #     fwd = self.front_pos
        #     cell = self.grid.get(*fwd)
        #     # if we just opened a door this step
        #     if isinstance(cell, Door) and cell.is_open:
        #         reward += 0.05

        # # 4) Encourage exploration
        # if action == self.actions.forward:
        #     new_pos = tuple(self.agent_pos)
        #     if new_pos not in self.visited:
        #         reward += 0.02    # bonus for exploring new cell
        #     else:
        #         reward -= 0.01    # small penalty for backtracking
        #     self.visited.add(new_pos)
        return obs, reward, term, trunc, info
    

# register to gymnasium. Must import this file first to trigger registration!!
from gymnasium.envs.registration import register

register(
    id="MiniGrid-DynamicBSP-v0",
    entry_point="PCGEnv:DynamicBSPMiniGridEnv",
    #max_episode_steps= 4096,
    kwargs={
        # any defaults
        "world_size":    (16, 16),
        "room_count":    2,
        "goal_count":    1,
        "barrier_count": 1,
        "lava_count":    2,
        "lava_length":   7,
        "min_room_size": 5,
    },
)
