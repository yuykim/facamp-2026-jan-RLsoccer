import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(SAVE_DIR, exist_ok=True)

env = football_env.create_environment(
    env_name="5_vs_5",
    render=False,
    write_video=False,
    representation="simple115v2",
    rewards="scoring,checkpoint"
)

# 체크포인트 콜백 설정 (50만 번마다 SAVE_DIR에 저장)
checkpoint_callback = CheckpointCallback(
    save_freq=500,
    save_path=SAVE_DIR,
    name_prefix="soccer_base"
)

obs = env.reset()

model = PPO("MlpPolicy", env, verbose=1)

print(f"학습 시작... 모델은 '{SAVE_DIR}' 폴더에 저장됩니다.")
model.learn(total_timesteps=1_000)

# 최종 모델 저장
final_path = os.path.join(SAVE_DIR, "final_model.zip")
model.save(final_path)

env.close()
print(f"학습 완료! 최종 모델 저장 위치: {final_path}")
