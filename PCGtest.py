
from minigrid.manual_control import ManualControl
from PCGEnv import DynamicBSPMiniGridEnv

# ──────────────────────────────────────────────────────────────────────────────
#  test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = DynamicBSPMiniGridEnv(
        world_size=(30,30),
        room_count=10,
        goal_count=1,
        barrier_count= 50,
        lava_count = 10,
        lava_length= 4,  
        render_mode="human"
    )

    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    obs, _ = env.reset()
    input("Press ENTER to exit…")
    env.close()
