from nash_env import NashEnv

env = NashEnv(seed=123)
obs, info = env.reset()
print("Game:", info["game_key"])
print("Description:", obs["description"][:80], "...")
action = (0,0)
_, reward, *_ = env.step(action)
print("Tried action:", action, "Reward:", reward)
