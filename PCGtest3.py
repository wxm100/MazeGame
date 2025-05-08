import imageio
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
import PCGEnv
from curriculum import Curriculums

models_path="./"

for stage in range (4,4):
    cfg= Curriculums[stage-1]
    print(f"evaluate {cfg}")
    eval_env = ImgObsWrapper(
        gym.make(
            "MiniGrid-DynamicBSP-v0",
            goal_count=1,
            lava_length=7,
            min_room_size=5,
            render_mode="rgb_array",
            **cfg,
        )
    )

    model = PPO.load(models_path + f"ppo_pcg_stage{stage}.zip", env=eval_env)

    frames = []
    obs, _ = eval_env.reset()
    done, truncated = False, False

    while not (done or truncated):
        frame = eval_env.render() 
        frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)

    # Save GIF
    imageio.mimsave(models_path + f"eval_stage{stage}.gif", frames, fps=5)
    print(f"Saved rollout to eval_stage{stage}.gif")
