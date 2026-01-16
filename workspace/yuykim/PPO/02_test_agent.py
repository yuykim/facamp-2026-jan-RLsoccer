import gfootball.env as football_env
from stable_baselines3 import PPO
import os
import utils

def main():
    # 1. 경로 및 환경 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 앞서 01_train.py에서 저장한 model 폴더 내의 최종 모델 경로
    model_path = os.path.join(BASE_DIR, "custom_model", "custom_retrained_5500000_steps.zip")

    env = football_env.create_environment(
        env_name="5_vs_5", # 학습한 시나리오와 동일하게 설정
        render=True,       # 화면 출력
        write_video=False,  # 자체 영상 저장 기능 활성화
        representation="simple115v2",
        rewards="scoring,checkpoint"
    )

    if os.path.exists(model_path):
        print(f"model: {model_path}")
        model = PPO.load(model_path, env=env, device="cpu")
    else:
        print(f"error: {model_path} no model.")
        return
    
    obs = env.reset()
    done = False
    max_steps = 3001 
    total_reward = 0
    t = 0

    while (not done) and (t < max_steps):

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)


        #frame = env.render(mode="rgb_array")
        #if frame is not None:
        #    utils.save_frame(frame, t)


        if reward != 0:
            print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)
        t += 1

        total_reward += reward

    env.close()

    print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)
    print("Total reward:", total_reward)


if __name__ == "__main__":

    #utils.cleanup()

    main()

    #utils.make_video()
