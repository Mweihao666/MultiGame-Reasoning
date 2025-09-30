import gymnasium as gym
import numpy as np
from ragen.env.base import BaseDiscreteActionEnv
from .config import NashEnvConfig

INIT_PROMPT = """You are playing a 2-action matrix game. Goal: choose a Nash equilibrium action.
Rules:
1) There are 2 actions: {name_a} and {name_b}
2) Payoff matrices for both players are hidden.
3) Your task is to choose an action that belongs to a Nash equilibrium.
"""

class NashEnv(BaseDiscreteActionEnv, gym.Env):
    """
    NashEnv：接口完全对齐 bandit
    - reset(seed) -> obs(str)
    - step(action) -> (obs: str, reward: int, done: bool, info: dict)
    - render() -> str
    - close() -> None
    """

    def __init__(self, config=None):
        BaseDiscreteActionEnv.__init__(self)
        self.config = config if config is not None else NashEnvConfig()

        # 与 bandit 一致：两个动作，索引从 action_space_start 开始
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(2, start=self.config.action_space_start)

        self.lo_action_name = self.config.lo_action_name
        self.hi_action_name = self.config.hi_action_name

        # payoff 矩阵（若未指定，给个默认博弈）
        if self.config.payoff_matrix_p1 is None:
            self.config.payoff_matrix_p1 = [[1, 0], [0, 1]]  # 协调博弈
        if self.config.payoff_matrix_p2 is None:
            self.config.payoff_matrix_p2 = [[1, 0], [0, 1]]

        self.render_cache = None
        assert self.config.render_mode == "text"

    def _randomize_actions(self):
        start = self.config.action_space_start
        if self.np_random.random() < 0.5:
            self.ACTION_LOOKUP = {
                start: self.lo_action_name,
                start + 1: self.hi_action_name,
            }
        else:
            self.ACTION_LOOKUP = {
                start: self.hi_action_name,
                start + 1: self.lo_action_name,
            }
        self.ARM_IDX_TO_NAME = dict(self.ACTION_LOOKUP)
        self.NAME_TO_ARM_IDX = {v: k for k, v in self.ARM_IDX_TO_NAME.items()}
        self.config.action_lookup = dict(self.ACTION_LOOKUP)

    def _find_nash_equilibria(self):
        """返回玩家1的所有纯策略纳什均衡索引"""
        nash_actions = []
        payoff1 = np.array(self.config.payoff_matrix_p1)
        payoff2 = np.array(self.config.payoff_matrix_p2)

        for a1 in range(2):  # 玩家1动作
            for a2 in range(2):  # 玩家2动作
                # 玩家1是否最佳回应
                best_resp1 = payoff1[:, a2].max()
                cond1 = payoff1[a1, a2] == best_resp1
                # 玩家2是否最佳回应
                best_resp2 = payoff2[a1, :].max()
                cond2 = payoff2[a1, a2] == best_resp2
                if cond1 and cond2:
                    nash_actions.append(a1)
        return nash_actions

    def reset(self, seed=None, mode=None):
        gym.Env.reset(self, seed=seed)
        self._randomize_actions()

        pos1 = self.config.action_space_start
        pos2 = pos1 + 1
        name1 = self.ARM_IDX_TO_NAME[pos1]
        name2 = self.ARM_IDX_TO_NAME[pos2]

        self.nash_actions = self._find_nash_equilibria()

        self.render_cache = INIT_PROMPT.format(name_a=name1, name_b=name2)
        return self.render_cache  

    def compute_reward(self, action: int) -> int:
        # 判定玩家1动作是否属于纳什均衡
        # 将索引映射回 {0,1}
        logical_idx = action - self.config.action_space_start
        return int(logical_idx in self.nash_actions)

    def step(self, action: int):
        assert action in self.ACTION_LOOKUP, f"Invalid action: {action}"

        reward = self.compute_reward(action)
        action_name = self.ARM_IDX_TO_NAME[action]

        next_obs = f"{action_name}: {reward} points"
        self.render_cache = next_obs

        done = True
        info = {
            "action_is_effective": True, 
            "action_is_valid": True,
            "success": bool(reward == 1),  # 这里表示是否选中纳什均衡
        }
        return next_obs, reward, done, info

    def get_all_actions(self):
        return [self.ACTION_SPACE.start, self.ACTION_SPACE.start + 1]

    def render(self):
        return self.render_cache

    def close(self):
        self.render_cache = None
