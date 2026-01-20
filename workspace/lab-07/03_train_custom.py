import gfootball.env as football_env
from stable_baselines3 import PPO
from custom_reward_good import CustomReward


env = football_env.create_environment(
    env_name="academy_run_to_score_with_keeper",
    render=False,
    write_video=False,
    representation="simple115v2",
    rewards="scoring",
)

env = CustomReward(env)

obs = env.reset()

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=50_000)

model.save("model-custom-reward.zip")

env.close()
