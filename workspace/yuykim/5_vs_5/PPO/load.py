import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
import os

def make_env():
    env = football_env.create_environment(
        env_name="5_vs_5",
        representation="simple115v2",
        rewards="scoring,checkpoint",
        render=False
    )
    return env

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(BASE_DIR, "model")
    
    # [수정] 마지막으로 저장된 체크포인트 파일 이름을 정확히 입력하세요.
    # 예: model 폴더 안에 있는 가장 숫자가 높은 파일
    LOAD_MODEL_NAME = "soccer_base_7500000_steps.zip" 
    LOAD_MODEL_PATH = os.path.join(SAVE_DIR, LOAD_MODEL_NAME)

    num_cpu = 8 
    env = make_vec_env(make_env, n_envs=num_cpu)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(500000 // num_cpu, 1),
        save_path=SAVE_DIR,
        name_prefix="soccer_base_resume" # 이어서 한다는 표시
    )

    # [수정] 기존 모델 불러오기
    if os.path.exists(LOAD_MODEL_PATH):
        print(f"[{LOAD_MODEL_NAME}]")
        # env를 명시적으로 전달해야 이어서 학습이 가능합니다.
        model = PPO.load(LOAD_MODEL_PATH, env=env, device="cpu") 
    else:
        # 파일이 없을 경우를 대비한 방어 코드 (새로 시작하거나 종료)
        exit()

    # [수정] 학습 시작
    # reset_num_timesteps=False 로 설정해야 로그상 스텝 수가 0이 아닌 750만부터 시작합니다.
    model.learn(
        total_timesteps=10_000_000, 
        callback=checkpoint_callback,
        reset_num_timesteps=False 
    )

    model.save(os.path.join(SAVE_DIR, "final_model_complete"))
    env.close()