import os
import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from custom_reward import CustomReward 

# 1. 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "custom_model")
LOAD_MODEL_PATH = os.path.join(SAVE_DIR, "soccer_base_5000000_steps.zip")

os.makedirs(SAVE_DIR, exist_ok=True)

# 2. 환경 생성 함수 (정규화 없이 커스텀 리워드만 적용)
def make_env():
    env = football_env.create_environment(
        env_name="5_vs_5",
        render=False,
        representation="simple115v2",
        rewards="scoring,checkpoint"
    )
    # 정규화 대신 커스텀 리워드로 직접 보상 체계 조절
    env = CustomReward(env)
    return env


env = make_env()

# 4. 체크포인트 콜백
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=SAVE_DIR,
    name_prefix="custom_retrained"
)

# 5. 기존 모델 불러오기
if os.path.exists(LOAD_MODEL_PATH):
    print(f"Loading existing model: {LOAD_MODEL_PATH}")
    # 정규화(VecNormalize) 없이 모델만 로드
    model = PPO.load(LOAD_MODEL_PATH, env=env, device="cpu", verbose=1)
    
    # [팁] 정규화를 안 쓸 경우, 학습률을 평소보다 낮게 설정하면 훨씬 안정적입니다.
    # model.learning_rate = 0.0001 
else:
    print(f"Model file not found. Starting from scratch.")
    model = PPO("MlpPolicy", env, verbose=1)

# 6. 학습 시작
print(f"Start training without Normalization... Checkpoints in: '{SAVE_DIR}'")
model.learn(total_timesteps=500_000, callback=checkpoint_callback, reset_num_timesteps=False)

# 7. 최종 모델 저장
final_path = os.path.join(SAVE_DIR, "final_model_no_norm.zip")
model.save(final_path)

env.close()
print(f"Training Complete! Final model saved at: {final_path}")