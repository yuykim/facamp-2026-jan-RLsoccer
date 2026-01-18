import sys

sys.path.append("..")
import utils
import gfootball.env as football_env
from gfootball.env import football_action_set
import pygame


## Movement actions
# `action_left` = 1, run to the left, sticky action.
# `action_top_left` = 2, run to the top-left, sticky action.
# `action_top` = 3, run to the top, sticky action.
# `action_top_right` = 4, run to the top-right, sticky action.
# `action_right` = 5, run to the right, sticky action.
# `action_bottom_right` = 6, run to the bottom-right, sticky action.
# `action_bottom` = 7, run to the bottom, sticky action.
# `action_bottom_left` = 8, run to the bottom-left, sticky action.

## Passing / Shooting
# `action_long_pass` = 9, perform a long pass to the player on your team. Player to pass the ball to is auto-determined based on the movement direction.
# `action_high_pass` = 10, perform a high pass, similar to `action_long_pass`.
# `action_short_pass` = 11, perform a short pass, similar to `action_long_pass`.
# `action_shot` = 12, perform a shot, always in the direction of the opponent's goal.

## Other actions
# `action_sprint` = 13, start sprinting, sticky action. Player moves faster, but has worse ball handling.
# `action_release_direction` = 14, reset current movement direction.
# `action_release_sprint` = 15, stop sprinting.
# `action_sliding` = 16, perform a slide (effective when not having a ball).
# `action_dribble` = 17, start dribbling (effective when having a ball), sticky action. Player moves slower, but it is harder to take over the ball from him.
# `action_release_dribble` = 18, stop dribbling.

# `sticky_actions` - 10-elements vectors of 0s or 1s denoting whether corresponding action is active:
# `0` - `action_left`
# `1` - `action_top_left`
# `2` - `action_top`
# `3` - `action_top_right`
# `4` - `action_right`
# `5` - `action_bottom_right`
# `6` - `action_bottom`
# `7` - `action_bottom_left`
# `8` - `action_sprint`
# `9` - `action_dribble`


def my_policy(obs, step):
    # unwrap (we're using the raw representation)
    obs = obs[0]

    # index of the player who owns the ball
    player = obs["ball_owned_player"]
    # player position
    player_x, player_y = obs["left_team"][player]

    # run to the top
    action = football_action_set.action_top

    print(
        "Step:", step, "Action:", action, f"Player Position: x={player_x} y={player_y}"
    )

    return action


def run_scenario():

    env = football_env.create_environment(
        env_name="academy_empty_goal_close",
        representation="raw",
        render=True,
        write_video=False,
        logdir="log",
        write_goal_dumps=True,
        write_full_episode_dumps=True,
        rewards="scoring,checkpoints",
    )

    obs = env.reset()

    done = False
    t = 0

    while not done:

        pygame.event.pump()

        action = my_policy(obs, t)
        obs, reward, done, info = env.step(action)

        frame = env.render(mode="rgb_array")
        utils.save_frame(frame, t)

        t += 1

    print("Step:", t, "Done:", done)

    env.close()

    print("Finished", t)


if __name__ == "__main__":

    pygame.init()

    utils.cleanup()

    run_scenario()

    utils.make_video()
