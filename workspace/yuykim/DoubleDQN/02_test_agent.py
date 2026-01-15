import gfootball.env as football_env
from double_dqn import DoubleDQN
import utils
import os


def main():

    env = football_env.create_environment(
        env_name="academy_run_to_score_with_keeper",
        render=False,
        write_video=False,
        representation="simple115v2",
        rewards="scoring",
    )

    obs = env.reset()


    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "DoubleDQN_model", "double_dqn_5v5_500k.zip")

    model = DoubleDQN.load(model_path, env=env, device="cpu")

    done = False
    max_steps = 500
    total_reward = 0
    t = 0

    while (not done) and (t < max_steps):

        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        frame = env.render(mode="rgb_array")
        if frame is not None:
            utils.save_frame(frame, t)

        if t % 20 == 0:
            print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)
        t += 1

        total_reward += reward

    env.close()

    print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)
    print("Total reward:", total_reward)


if __name__ == "__main__":

    utils.cleanup()

    main()

    utils.make_video()
