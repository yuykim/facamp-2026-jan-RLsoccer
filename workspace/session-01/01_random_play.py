import os
import numpy as np
from PIL import Image
import gfootball.env as football_env

OUT_DIR = "frames"

os.makedirs(OUT_DIR, exist_ok=True)

env = football_env.create_environment(
    env_name="11_vs_11_stochastic",
    render=True,
    write_video=False
)

obs = env.reset()

done = False
max_steps = 300
t = 0

while (not done) and (t < max_steps):

    action = env.action_space.sample()  # random play
    obs, reward, done, info = env.step(action)

    frame = env.render(mode="rgb_array")
    if frame is not None:
        Image.fromarray(frame).save(f"{OUT_DIR}/{t:05d}.png")

    if t % 10 == 0:
        print(f"Step: {t}/{max_steps}", "Done:", done)
    t += 1

env.close()

print("Finished", t)
