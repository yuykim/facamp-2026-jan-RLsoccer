import gfootball.env as football_env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import os

from custom_reward import CustomReward

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 1. 환경 생성 (단일 환경)
def create_single_env():
    env = football_env.create_environment(
        env_name="5_vs_5",
        representation="simple115v2",
        rewards="scoring,checkpoint",
        render=False
    )
    # 2. 직접 만든 보상 래퍼 적용
    env = CustomReward(env)
    # 3. 자동 정규화를 위해 VecEnv로 감싸기
    env = DummyVecEnv([lambda: env])
    # 4. 리워드 정규화 적용 (학습 시에만 사용)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.)
    return env

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(BASE_DIR, "model_double_dqn")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 환경 생성
    env = create_single_env()

    # 2. 체크포인트 콜백 (단일 환경이므로 실제 스텝 수대로 저장)
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=SAVE_DIR,
        name_prefix="soccer_dqn"
    )

    # 3. Double DQN 모델 정의
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cpu",             # GPU가 있다면 "cuda"로 변경 가능
        learning_rate=1e-4,       # 학부 수준에서 가장 무난한 시작점
        buffer_size=100000,        # 메모리 사양에 따라 조절 (단일 환경이므로 5만~10만 적당)
        learning_starts=5000,     # 초반에 데이터가 어느 정도 쌓인 후 학습 시작
        batch_size=32,            # 업데이트 시 한 번에 꺼낼 샘플 양
        tau=1.0,                  # 1.0이면 Hard Update (주기마다 타겟 네트워크를 통째로 복사)
        target_update_interval=1000, # 타겟 네트워크를 업데이트하는 간격(스텝 단위)
        train_freq=4,             # 4스텝마다 한 번씩 학습 진행
        gradient_steps=1,         # 학습 빈도당 경사하강법 수행 횟수
        exploration_fraction=0.1,  # 전체 학습의 10% 구간 동안 epsilon 감소
        exploration_final_eps=0.05, # 최종적으로 5% 확률은 탐험에 사용
        # double_q=True             # ⭐ Double DQN 활성화
    )

    # 4. 학습 시작
    print("Starting Double DQN training in a single environment...")
    model.learn(total_timesteps=5_000_000, callback=checkpoint_callback)

    # 최종 저장
    model.save(os.path.join(SAVE_DIR, "final_dqn_single_model"))
    env.save(os.path.join(SAVE_DIR, "stats.pkl"))
    env.close()