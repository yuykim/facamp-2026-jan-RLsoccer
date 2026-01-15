import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from custom_reward import CustomReward
import os
import torch

# 03_use_pre_trained_model.py 내의 환경 생성 함수 수정
def make_env():
    def _init():
        env = football_env.create_environment(
            env_name="5_vs_5",              # 기존 공식 5대5 시나리오 사용
            representation="simple115v2",
            rewards="scoring",
            render=False,
            # [여기가 핵심!] 난이도 오버라이드 설정
            other_config_options={
                'right_team_difficulty': 1.0, # 상대 팀을 최강으로
                'left_team_difficulty': 0.7   # 우리 팀 동료는 적당히 똑똑하게
            }
        )
        env = CustomReward(env) 
        return env
    return _init

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(BASE_DIR, "model")
    os.makedirs(SAVE_DIR, exist_ok=True)

    LOAD_MODEL_PATH = os.path.join(SAVE_DIR, "soccer_base_5000000_steps.zip")

    # M1 맥북을 위한 환경 개수 (2개)
    num_cpu = 2 
    env = make_vec_env(make_env(), n_envs=num_cpu)

    # 1. 현재 환경에 맞는 '새 모델'을 먼저 만듭니다. (n_envs=2로 자동 설정됨)
    model = PPO("MlpPolicy", env, verbose=1, device="cpu", n_steps=2048, batch_size=64)

    # 2. 기존 모델에서 가중치(파라미터)만 로드하여 덮어씌웁니다.
    if os.path.exists(LOAD_MODEL_PATH):
        print(f"--- Loading weights from: {LOAD_MODEL_PATH} ---")
        
        # 모델 파일에서 파라미터만 추출
        temp_model = PPO.load(LOAD_MODEL_PATH)
        parameters = temp_model.get_parameters()
        
        # 새 모델에 주입
        model.set_parameters(parameters)
        print("--- Success: Weights transferred to new model with 2 Envs ---")
    else:
        print("--- No model found. Starting from scratch ---")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000000 // num_cpu, 1),
        save_path=SAVE_DIR,
        name_prefix="hard_5_vs_5"
    )

    print(f"[{num_cpu} Envs] Training Start!")
    
    # 이제 모델 자체가 n_envs=2로 생성되었으므로 에러가 날 수 없습니다.
    model.learn(total_timesteps=5_000_000, callback=checkpoint_callback)

    model.save(os.path.join(SAVE_DIR, "final_model_custom_500k"))
    env.close()