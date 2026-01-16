import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os
from custom_reward import CustomReward

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(SAVE_DIR, exist_ok=True)

env = football_env.create_environment(
    env_name="academy_run_to_score_with_keeper",
    render=False,
    write_video=False,
    representation="simple115v2",
    # rewards="scoring,checkpoint"
    rewards="scoring"
)

env = CustomReward(env)

# 체크포인트 콜백 설정
checkpoint_callback = CheckpointCallback(
    save_freq=20_000,
    save_path=SAVE_DIR,
    name_prefix="soccer_base"
)

obs = env.reset()

model = PPO("MlpPolicy", env, verbose=1)

print(f"start....'{SAVE_DIR}' <-- save this dir")
model.learn(total_timesteps=50_000)

# 최종 모델 저장
final_path = os.path.join(SAVE_DIR, "final_model.zip")
model.save(final_path)

env.close()
print(f"complete save_path: {final_path}")
