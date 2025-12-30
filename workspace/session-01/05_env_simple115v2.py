from pprint import pprint
import gfootball.env as football_env

# simple115v2 representation
env = football_env.create_environment(
    env_name="11_vs_11_competition", representation="simple115v2", render=False
)

obs = env.reset()

print("gfootball eenvironment reset OK")
print("action space:", env.action_space)

print(obs)

print(len(obs))

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
obs_dict["left_team"] = obs[0:22]
obs_dict["left_team_direction"] = obs[22:44]
obs_dict["right_team"] = obs[44:66]
obs_dict["right_team_direction"] = obs[66:88]
obs_dict["ball"] = obs[88:91]
obs_dict["ball_direction"] = obs[91:94]
obs_dict["ball_ownership"] = obs[94:97]
obs_dict["active_player"] = obs[97:108]
obs_dict["game_mode"] = obs[108:115]

pprint(obs_dict)

env.close()
