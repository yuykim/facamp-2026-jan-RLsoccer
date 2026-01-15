import gfootball.env as football_env
from stable_baselines3 import DQN
import os
import utils as utils

def main():
    # 1. 경로 및 환경 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "model_dqn", "soccer_dqn_600000_steps.zip")

    env = football_env.create_environment(
        env_name="5_vs_5",
        render=True,
        write_video=False,
        representation="simple115v2",
    )

    if os.path.exists(model_path):
        print(f"model: {model_path}")
        model = DQN.load(model_path, env=env, device="cpu")
    else:
        print(f"error: {model_path} no model.")
        return

    # reset() 호환 (gym/gymnasium)
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    done = False
    max_steps = 1000
    total_reward = 0
    t = 0

    while (not done) and (t < max_steps):

        action, _ = model.predict(obs, deterministic=True)

        # step() 호환 (gym/gymnasium)
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out

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
