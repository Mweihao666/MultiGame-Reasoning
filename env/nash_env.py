import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import nashpy as nash


class NashEnv(gym.Env):
    """
    一个矩阵博弈环境，接口模仿 OpenAI Gym。
    reset() -> 返回随机生成的博弈场景描述（prompt）
    step(action) -> 输入玩家的动作，返回 (state, reward, done, info)
    """

    def __init__(self):
        super(NashEnv, self).__init__()

        # 定义可用的 payoff 矩阵
        self.games = [
            {
                "name": "Prisoner's Dilemma",
                "A": np.array([[-1, -3], [0, -2]]),
                "B": np.array([[-1, 0], [-3, -2]])
            },
            {
                "name": "Coordination Game",
                "A": np.array([[2, 0], [0, 1]]),
                "B": np.array([[2, 0], [0, 1]])
            }
        ]

        # 动作空间：假设玩家只能选 0 或 1
        self.action_space = spaces.Discrete(2)
        # 观测空间：这里只是占位，主要返回 prompt
        self.observation_space = spaces.Discrete(1)

        self.current_game = None
        self.equilibria = None
        self.prompt = ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 随机抽一个博弈
        self.current_game = random.choice(self.games)
        A, B = self.current_game["A"], self.current_game["B"]

        # 用 nashpy 计算纳什均衡
        game = nash.Game(A, B)
        eqs = list(game.support_enumeration())
        self.equilibria = eqs

        # 生成多样化的 prompt
        descriptions = [
            f"这是一个{self.current_game['name']}。矩阵如下：{A.tolist()}, {B.tolist()}。请选择你的策略 (0 或 1)。",
            f"你现在参与一场博弈：{self.current_game['name']}。可选动作有两个 (0/1)，不同选择带来不同收益。",
            f"考虑以下博弈：{self.current_game['name']}，行玩家矩阵 = {A.tolist()}，列玩家矩阵 = {B.tolist()}。选择你的动作。"
        ]
        self.prompt = random.choice(descriptions)

        return self.prompt

    def step(self, action):
        if self.current_game is None:
            raise ValueError("必须先调用 reset()")

        # 检查 Nash 均衡的最优动作
        eq_actions = []
        for sigma_r, sigma_c in self.equilibria:
            best_action = np.argmax(sigma_r)  # 行玩家最优动作
            eq_actions.append(best_action)

        # reward 规则
        reward = 1 if action in eq_actions else -1

        # 单回合游戏：直接结束
        done = True
        info = {"equilibria": self.equilibria, "chosen_action": action}

        return self.prompt, reward, done, info
