import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
import os

# 1. 환경을 만드는 함수 정의
def make_env():
    env = football_env.create_environment(
        env_name="5_vs_5",
        representation="simple115v2",
        rewards="scoring,checkpoint",
        render=False
    )
    return env

if __name__ == "__main__":
    # 폴더 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(BASE_DIR, "model")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 2. 병렬 환경 설정 (n_envs를 CPU 코어 수에 맞춰 조절하세요)
    # 보통 본인 CPU 코어 수의 70~80% 정도를 추천합니다 (예: 8코어면 6~8)
    num_cpu = 8 
    env = make_vec_env(make_env, n_envs=num_cpu)

    # 3. 체크포인트 콜백 (전체 스텝 수 기준이므로 num_cpu로 나눠줌)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(500000 // num_cpu, 1),
        save_path=SAVE_DIR,
        name_prefix="soccer_base"
    )

    # 4. 모델 정의 (GPU가 있다면 device="cuda" 추가)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cuda", # GPU 있으면 자동으로 사용
        n_steps=2048,  # 한 번에 수집하는 데이터 양 조정
        batch_size=64  # 학습 속도 최적화
    )

    # 5. 학습 시작
    print(f"[{num_cpu}개 병렬 환경] 학습 시작... 목표 1,000만 스텝")
    model.learn(total_timesteps=10_000_000, callback=checkpoint_callback)

    # 최종 저장
    model.save(os.path.join(SAVE_DIR, "final_model"))
    env.close()