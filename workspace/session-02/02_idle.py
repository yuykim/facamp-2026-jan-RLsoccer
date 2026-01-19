import utils
import gfootball.env as football_env


def my_policy(step):
    return 12  # do shooting


def run_scenario():

    env = football_env.create_environment(
        env_name="academy_empty_goal_close",
        representation="raw",
        render=True,
        write_video=False,
        logdir="log",
        write_goal_dumps=True,
        write_full_episode_dumps=True,
    )

    obs = env.reset()

    done = False
    max_steps = 400
    t = 0

    while (not done) and (t < max_steps):

        action = my_policy(t)
        obs, reward, done, info = env.step(action)

        frame = env.render(mode="rgb_array")
        if frame is not None:
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
