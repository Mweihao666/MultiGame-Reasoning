from env.nash_env import NashEnv

if __name__ == "__main__":
    env = NashEnv()

    # 初始化环境
    prompt = env.reset()
    print("Prompt:", prompt)

    # 假设玩家选择动作 0
    state, reward, done, info = env.step(0)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)
