import gfootball.env as football_env
from stable_baselines3 import PPO
from custom_reward import CustomReward
import utils
import os

SCENARIO_NAME = "academy_corner"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.zip")

def main():
    # 모델 파일 존재 여부 확인
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Can't find model file at ({MODEL_PATH})")
        raise SystemExit(1)

    env = football_env.create_environment(
        env_name=SCENARIO_NAME,
        render=True,
        write_video=False,
        representation="simple115v2",
        rewards="scoring",
    )

    env = CustomReward(env)

    obs = env.reset()

    model = PPO.load(MODEL_PATH, device="cpu")

    done = False
    max_steps = 500
    total_reward = 0
    max_episodes = 1  # 총 실행할 에피소드 횟수
    t = 0


    for episode in range(max_episodes):
        obs = env.reset()  # 에피소드 시작 시 환경 초기화
        done = False
        t = 0
        episode_reward = 0

        print(f"\n--- Episode {episode + 1} Start ---")

        while not done and t < max_steps:
            action, _ = model.predict(obs, deterministic=True) # 결정적 행동 선택
            obs, reward, done, info = env.step(action)


            #frame = env.render(mode="rgb_array")
            #if frame is not None:
            #    utils.save_frame(frame, t)
            
            episode_reward += reward
            
            if t % 50 == 0:
                print(f"Step: {t}/{max_steps} | Current Reward: {reward:.2f}")
            
            t += 1

        print(f"Episode {episode + 1} Finished | Total Reward: {episode_reward:.2f} | Steps: {t}")

    env.close()
    
    print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)
    print("Total reward:", total_reward)


if __name__ == "__main__":

    #utils.cleanup()

    main()

    #utils.make_video()
