import gfootball.env as football_env
from double_dqn import DoubleDQN

def make_env():
    return football_env.create_environment(
        env_name="5_vs_5",
        render=False,
        write_video=False,
        representation="simple115v2",
        rewards="scoring",  # 일단 scoring만 추천 (평가/해석 쉬움)
    )

env = make_env()

model = DoubleDQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=200_000,
    learning_starts=10_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=10_000,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1,
)

model.learn(total_timesteps=500_000)
model.save("double_dqn_5v5_scoring_500k")
env.close()
