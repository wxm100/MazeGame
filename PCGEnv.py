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
        self.action_space = Discrete(self.actions.forward + 1)
        self.obstacles: List[Ball] = []

    def _gen_grid(self, width: int, height: int):
        # 1) Outer walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # 2) Iterative BSP splitting
        regions: List[Tuple[int,int,int,int]] = [(0, 0, width, height)]
        splits = self.room_count - 1

        while splits > 0:
            # find splittable regions
            cands = []
            for idx, (x, y, w, h) in enumerate(regions):
                can_v = (w >= 2*self.min_room_size+1 and h >= 3)
                can_h = (h >= 2*self.min_room_size+1 and w >= 3)
                if can_v or can_h:
                    cands.append((idx, x, y, w, h, can_v, can_h))
            if not cands:
                break

            # pick the largest area region
            idx, x, y, w, h, can_v, can_h = max(cands, key=lambda t: t[3]*t[4])
            regions.pop(idx)

            if can_v and (not can_h or w >= h):
                # vertical split
                cut = random.randint(self.min_room_size, w - self.min_room_size - 1)

                # left/right are floor, but above/below are still walls
                door_rows = [
                    j
                    for j in range(y+1, y+h-1)
                    if (
                        self.grid.get(x+cut-1, j) is None and
                        self.grid.get(x+cut+1, j) is None and
                        isinstance(self.grid.get(x+cut, j-1), Wall) and
                        isinstance(self.grid.get(x+cut, j+1), Wall)
                    )
                ]
                if door_rows:
                    door_j = random.choice(door_rows)
                else:
                    door_j = random.randint(y+1, y+h-2)

                # draw that split line
                for j in range(y, y+h):
                    obj = Door(random.choice(COLOR_NAMES)) if j == door_j else Wall()
                    self.grid.set(x+cut, j, obj)

                regions.append((x,        y, cut,        h))
                regions.append((x+cut+1, y, w-cut-1, h))

            else:
                # horizontal split
                cut = random.randint(self.min_room_size, h - self.min_room_size - 1)

                # up/down are floor, but left/right are still walls
                door_cols = [
                    i
                    for i in range(x+1, x+w-1)
                    if (
                        self.grid.get(i, y+cut-1) is None and
                        self.grid.get(i, y+cut+1) is None and
                        isinstance(self.grid.get(i-1, y+cut), Wall) and
                        isinstance(self.grid.get(i+1, y+cut), Wall)
                    )
                ]
                if door_cols:
                    door_i = random.choice(door_cols)
                else:
                    door_i = random.randint(x+1, x+w-2)

                for i in range(x, x+w):
                    obj = Door(random.choice(COLOR_NAMES)) if i == door_i else Wall()
                    self.grid.set(i, y+cut, obj)

                regions.append((x, y,       w, cut))
                regions.append((x, y+cut+1, w, h-cut-1))

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
            (i,j)
            for i in range(1, width-1)
            for j in range(1, height-1)
            if self.grid.get(i,j) is None
        ]
        random.shuffle(floor)
        for _ in range(min(self.goal_count, len(floor))):
            gx,gy = floor.pop()
            self.grid.set(gx, gy, Goal())

        # 6) Moving Balls
        self.obstacles = []
        random.shuffle(floor)
        for _ in range(min(self.n_obstacles, len(floor))):
            ox,oy = floor.pop()
            b = Ball()
            self.grid.set(ox, oy, b)
            b.cur_pos = (ox, oy)
            self.obstacles.append(b)

        # 7) Agent
        random.shuffle(floor)
        if floor:
            self.agent_pos = floor.pop()
        else:
            fx,fy,fw,fh = regions[0]
            self.agent_pos = (fx+fw//2, fy+fh//2)
        self.agent_dir = random.randint(0, 3)

    def step(self, action: int):
        # move Balls
        for b in self.obstacles:
            ox,oy = b.cur_pos
            self.grid.set(ox,oy,None)
            top = (max(ox-1,1), max(oy-1,1))
            try:
                newp = self.place_obj(b, top=top, size=(3,3), max_tries=100)
                b.cur_pos = newp
            except:
                self.grid.set(ox,oy,b)
                b.cur_pos = (ox,oy)

        obs, reward, term, trunc, info = super().step(action)
        if action == self.actions.forward:
            front = self.grid.get(*self.front_pos)
            if front and front.type=="ball":
                reward = -1
                term = True
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
