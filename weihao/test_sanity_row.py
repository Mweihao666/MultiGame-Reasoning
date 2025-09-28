from nashenv.env import NashEnv
import numpy as np

def main():
    env = NashEnv(seed=42, eval_mode="row")
    total_reward = 0
    for ep in range(20):
        obs, info = env.reset()
        # 在 row 模式下，动作只需要给 row index
        i = np.random.randint(0, len(obs["rows"]))
        _, reward, done, info2 = env.step(i)
        total_reward += reward
        print(f"Episode {ep+1:02d} | Game={info['game_key']:<18s} "
              f"| Row={i} | Reward={reward} | Done={done} "
              f"| Reason={info2['reason']}")
    print("="*80)
    print(f"Total reward over 20 episodes: {total_reward}")

if __name__ == "__main__":
    main()
