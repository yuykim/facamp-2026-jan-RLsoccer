import sys

sys.path.append("..")
import utils
import gfootball.env as football_env
from gfootball.env import football_action_set


def my_policy(step):
    if step < 20:
        # move right
        return football_action_set.action_right
    else:
        # reset current movement direction
        return football_action_set.action_release_direction


def run_scenario():

    env = football_env.create_environment(
        env_name="academy_empty_goal_close", representation="raw"
    )

    obs = env.reset()

    done = False
    max_steps = 400
    t = 0

    while (not done) and (t < max_steps):

        action = my_policy(t)
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
