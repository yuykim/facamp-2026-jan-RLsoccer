import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from custom_reward import CustomReward
import os

# 1. 환경을 만드는 함수 정의
def make_env():
    def _init():
        env = football_env.create_environment(
            env_name="5_vs_5",
            representation="simple115v2",
            rewards="scoring,checkpoint",
            render=False
        )
        env = CustomReward(env) 
        return env
    return _init

if __name__ == "__main__":
    # 폴더 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(BASE_DIR, "model")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 기존 모델 로드
    LOAD_MODEL_PATH = os.path.join(SAVE_DIR, "soccer_base_5000000_steps.zip")

    # 2. 병렬 환경 설정 (n_envs를 CPU 코어 수에 맞춰 조절하세요)
    # 보통 본인 CPU 코어 수의 70~80% 정도를 추천합니다 (예: 8코어면 6~8)
    num_cpu = 2 
    env = make_vec_env(make_env, n_envs=num_cpu)

    # 3. 체크포인트 콜백 (전체 스텝 수 기준이므로 num_cpu로 나눠줌)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(500000 // num_cpu, 1),
        save_path=SAVE_DIR,
        name_prefix="custom"
    )

    # 모델 정의 또는 로드
    if os.path.exists(LOAD_MODEL_PATH):
        print("start")
        model = PPO.load(LOAD_MODEL_PATH, env=env)
    else:
        print("No model")

    print(f"[{num_cpu} Envs]")
    model.learn(total_timesteps=10_000_000, callback=checkpoint_callback)

    model.save(os.path.join(SAVE_DIR, "final_model_custom"))
    env.close()