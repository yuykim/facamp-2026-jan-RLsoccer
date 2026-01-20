import gym


class CustomReward(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.w_shot = 0.02

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):

        # action = env.action_space.sample()  # random play
        obs, env_reward, done, info = self.env.step(action)

        obs_dict = self.obs_to_dict(obs)

        ball_x, ball_y = obs_dict["ball"][0], obs_dict["ball"][1]

        # reward if a shot is taken near the goal area
        near_goal = (ball_x > 0.75) and (abs(ball_y) < 0.25)
        took_shot = action == 12
        shot_bonus = 1.0 if (near_goal and took_shot) else 0.0

        custom_reward = self.w_shot * shot_bonus

        return obs, custom_reward, done, info

    def obs_to_dict(self, obs):
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

        return obs_dict
