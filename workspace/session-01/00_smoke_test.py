import gfootball.env as football_env

env = football_env.create_environment(env_name="11_vs_11_stochastic", render=False)

env.reset()

print("gfootball env reset OK")

env.close()
