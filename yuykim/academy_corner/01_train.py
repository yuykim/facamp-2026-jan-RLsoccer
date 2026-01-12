import gfootball.env as football_env
from stable_baselines3 import A2C
from custom_reward import CustomReward
import os

SCENARIO_NAME = "academy_corner"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
final_path = os.path.join(BASE_DIR, "model.zip")

env = football_env.create_environment(
    env_name=SCENARIO_NAME,
    render=False,
    write_video=False,
    representation="simple115v2",
    rewards="scoring",
)

env = CustomReward(env)

obs = env.reset()

# model = PPO("MlpPolicy", env, verbose=1)

model = A2C(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=7e-4,    # A2C는 PPO보다 조금 높은 학습률에서 잘 작동하는 경우가 많습니다.
    n_steps=5,             # 5걸음마다 바로 업데이트 (매우 빈번한 업데이트)
    ent_coef=0.01,         # 탐험을 유도하는 계수
    vf_coef=0.5,           # 가치 함수 손실 계수
)

model.learn(total_timesteps=100_000)

model.save(final_path)

env.close()
