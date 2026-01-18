import sys

sys.path.append("..")
import utils
import gfootball.env as football_env


def run_scenario():

    env = football_env.create_environment(
        env_name="academy_empty_goal_close", representation="raw"
    )

    obs = env.reset()

    done = False
    max_steps = 100
    t = 0

    while (not done) and (t < max_steps):

        action = env.action_space.sample()  # random play
        obs, reward, done, info = env.step(action)

        frame = env.render(mode="rgb_array")
        utils.save_frame(frame, t)

        if t % 10 == 0:
            print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)
        t += 1

    print(f"Step: {t}/{max_steps}", "Reward:", reward, "Done:", done)

    env.close()

    print("Finished", t)


if __name__ == "__main__":

    utils.cleanup()

    run_scenario()

    utils.make_video()
