# train_pcg_curriculum.py
import imageio
import gymnasium as gym
import os
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecTransposeImage, VecMonitor
)
from MinigridFeaturesExtractor import MinigridFeaturesExtractor
import PCGEnv# triggers the env registration
import numpy as np
from curriculum import Curriculums

# ----------------------------------------
# 1) Helper: build a vectorized, image‐obs env
# ----------------------------------------
def make_vec_env(cfg, n_envs=8):
    def _thunk():
        env = gym.make(
            "MiniGrid-DynamicBSP-v0",
            goal_count=1,
            lava_length=7,
            min_room_size=5,
            render_mode="rgb_array",
            **cfg
        )
        return ImgObsWrapper(env)
    venv = DummyVecEnv([_thunk for _ in range(n_envs)])
    venv = VecTransposeImage(venv)
    venv = VecMonitor(venv)
    return venv

# ----------------------------------------
# 2) PPO policy kwargs (custom CNN)
# ----------------------------------------
policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

# ------------------------------------------------
# 3) Curriculum training loop - include evaluation
# ------------------------------------------------
model = None
base_ckpt = "./ppo_pcg_stage4.zip"
timesteps_per_stage = 100_0000

for stage_idx, cfg in enumerate(Curriculums, start=1):
    print(f"\n=== Stage {stage_idx}/{len(Curriculums)}: {cfg} ===")
    venv = make_vec_env(cfg, n_envs=8)

    if model is None:
       # Either load from disk, or start new
        if base_ckpt and os.path.isfile(base_ckpt):
            model = PPO.load(base_ckpt, env=venv)
            print(f"Loaded checkpoint {base_ckpt}")
        else:
            model = PPO(
                "CnnPolicy",
                venv,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="./ppo_pcg_tensorboard/",
            )
            print("Started new PPO model")
    else:
        # Further stages: keep learned weights
        model.set_env(venv)
    

    model.learn(
        total_timesteps= min(timesteps_per_stage*1.5,10000000),
        reset_num_timesteps=False,
    )
    model.save(f"ppo_pcg_stage{stage_idx}")

    print(f"\nEvaluating stage{stage_idx} and recording GIF...")
    eval_env = ImgObsWrapper(
    gym.make(
        "MiniGrid-DynamicBSP-v0",
        goal_count=1,
        lava_length=4,
        min_room_size=5,
        render_mode="rgb_array",
        **cfg
        )
    )

    obs, _ = eval_env.reset()
    frames = []
    done = False

    while not done:
        frames.append(eval_env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = eval_env.step(action)
        done = done or trunc

    imageio.mimsave(f"ppo_pcg_curriculum_stage{stage_idx}.gif", frames, fps=5)

print("\n✅ Curriculum training complete.")
