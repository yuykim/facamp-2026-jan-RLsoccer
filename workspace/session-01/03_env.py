import gfootball.env as football_env

env = football_env.create_environment(
    env_name="11_vs_11_competition", render=False
)

obs = env.reset()

print("gfootball eenvironment reset OK")
print("action space:", env.action_space)

print(obs.shape)
print(obs)

env.close()
