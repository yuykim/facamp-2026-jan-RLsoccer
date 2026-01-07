from pprint import pprint
import gfootball.env as football_env

# simple115v2 representation with frame stacking
env = football_env.create_environment(
    env_name="11_vs_11_competition",
    representation="simple115v2",
    stacked=True,
    render=False,
)

obs = env.reset()

for t in range(50):

    action = env.action_space.sample()  # random play
    obs, reward, done, info = env.step(action)

print("gfootball eenvironment reset OK")
print("action space:", env.action_space)

print(obs)

# 4-frame temporal stacking is happening (460 = 115 * 4)
# [obs(t-3), obs(t-2), obs(t-1), obs(t)]
print(len(obs))

for n in range(len(obs) // 115):
    print(f"Frame {n}", "-" * 60)
    frame_obs = obs[n * 115 : (n + 1) * 115]

    # 22 - (x,y) coordinates of left team players
    # 22 - (x,y) direction of left team players
    # 22 - (x,y) coordinates of right team players
    # 22 - (x, y) direction of right team players
    # 3 - (x, y and z) - ball position
    # 3 - ball direction
    # 3 - one hot encoding of ball ownership (noone, left, right)
    # 11 - one hot encoding of which player is active
    # 7 - one hot encoding of `game_mode`

    obs_dict = {}
    obs_dict["left_team"] = frame_obs[0:22]
    obs_dict["left_team_direction"] = frame_obs[22:44]
    obs_dict["right_team"] = frame_obs[44:66]
    obs_dict["right_team_direction"] = frame_obs[66:88]
    obs_dict["ball"] = frame_obs[88:91]
    obs_dict["ball_direction"] = frame_obs[91:94]
    obs_dict["ball_ownership"] = frame_obs[94:97]
    obs_dict["active_player"] = frame_obs[97:108]
    obs_dict["game_mode"] = frame_obs[108:115]

    pprint(obs_dict)

env.close()
