# env/nash_env.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

# 可扩展的博弈注册表
# 每个博弈包含：
# - "name": 名称
# - "actions_row": 行玩家动作名
# - "actions_col": 列玩家动作名
# - "A": 行玩家收益矩阵 (numpy.ndarray, shape m x n)
# - "B": 列玩家收益矩阵 (numpy.ndarray, shape m x n)
# - "pure_ne": 纯策略纳什均衡列表（用索引对表示，如 [(0,0), (1,1)]）
# - "has_mixed": 是否存在（且仅有）混合均衡（比如 Matching Pennies、RPS）
GAMES_REGISTRY: Dict[str, Dict] = {
    # 1) 囚徒困境（你已有，可参考此格式）
    "PrisonersDilemma": {
        "name": "Prisoner's Dilemma",
        "actions_row": ["Cooperate", "Defect"],
        "actions_col": ["Cooperate", "Defect"],
        # 典型 PD 收益：T>R>P>S，取 (R=3, T=5, P=1, S=0)
        "A": np.array([[3, 0],
                       [5, 1]], dtype=float),
        "B": np.array([[3, 5],
                       [0, 1]], dtype=float),
        "pure_ne": [(1, 1)],  # (Defect, Defect)
        "has_mixed": False,
    },

    # 2) 协调博弈（你已有，可参考此格式）
    "Coordination": {
        "name": "Coordination Game",
        "actions_row": ["A", "B"],
        "actions_col": ["A", "B"],
        "A": np.array([[2, 0],
                       [0, 1]], dtype=float),
        "B": np.array([[2, 0],
                       [0, 1]], dtype=float),
        "pure_ne": [(0, 0), (1, 1)],
        "has_mixed": False,
    },

    # 3) 猎鹿博弈（Stag Hunt）
    # 常用参数：Stag/Stag 高回报；Hare/Hare 次优；不协调时一方亏
    "StagHunt": {
        "name": "Stag Hunt",
        "actions_row": ["Stag", "Hare"],
        "actions_col": ["Stag", "Hare"],
        "A": np.array([[4, 0],
                       [3, 2]], dtype=float),
        "B": np.array([[4, 3],
                       [0, 2]], dtype=float),
        "pure_ne": [(0, 0), (1, 1)],  # (Stag,Stag) 与 (Hare,Hare)
        "has_mixed": False,
    },

    # 4) 性别之战（Battle of the Sexes）
    # 两个协调均衡偏好不同；还存在一个混合均衡（此处只标纯均衡）
    "BattleOfSexes": {
        "name": "Battle of the Sexes",
        "actions_row": ["Ballet", "Football"],
        "actions_col": ["Ballet", "Football"],
        "A": np.array([[2, 0],
                       [0, 1]], dtype=float),
        "B": np.array([[1, 0],
                       [0, 2]], dtype=float),
        "pure_ne": [(0, 0), (1, 1)],
        "has_mixed": True,   # 有混合，但环境只判纯（可扩展）
    },

    # 5) Matching Pennies（只存在混合均衡）
    "MatchingPennies": {
        "name": "Matching Pennies",
        "actions_row": ["Heads", "Tails"],
        "actions_col": ["Heads", "Tails"],
        # 行玩家希望相同；列玩家希望不同
        "A": np.array([[1, -1],
                       [-1, 1]], dtype=float),
        "B": np.array([[-1, 1],
                       [1, -1]], dtype=float),
        "pure_ne": [],       # 无纯均衡
        "has_mixed": True,   # p=q=0.5 的混合均衡
    },

    # 6) 石头剪刀布（RPS，零和，只有均匀混合均衡）
    "RockPaperScissors": {
        "name": "Rock-Paper-Scissors",
        "actions_row": ["Rock", "Paper", "Scissors"],
        "actions_col": ["Rock", "Paper", "Scissors"],
        "A": np.array([[0, -1, 1],
                       [1,  0, -1],
                       [-1, 1, 0]], dtype=float),
        "B": None,  # 零和：B = -A
        "pure_ne": [],
        "has_mixed": True,   # 均匀(1/3,1/3,1/3)
    },

    # 7) 一个简单的零和 2x2（存在鞍点即纯均衡的情况也可自定）
    "ZeroSum2x2": {
        "name": "Zero-Sum (2x2)",
        "actions_row": ["Top", "Bottom"],
        "actions_col": ["Left", "Right"],
        "A": np.array([[1, -1],
                       [-1, 1]], dtype=float),
        "B": None,           # B = -A
        "pure_ne": [],       # 这个矩阵无纯 NE（与 MP 同构）
        "has_mixed": True,
    },
}

# 多样化表述模板（会在 reset() 随机挑选）
PROMPT_TEMPLATES: List[str] = [
    # 课堂定义式
    (
        "You are Row player in a normal-form game called <{name}>.\n"
        "Available actions (Row): {acts_row}\n"
        "Opponent (Column) actions: {acts_col}\n"
        "Payoff matrices (Row=A, Column=B):\nA=\n{A}\nB=\n{B}\n"
        "Return a joint action as (row_action, col_action)."
    ),
    # 生活化故事式
    (
        "场景：这是一个名为「{name}」的双人同时行动博弈。你控制“行玩家”，对手是“列玩家”。\n"
        "你的可选动作：{acts_row}；对手可选动作：{acts_col}。\n"
        "如下是双方收益矩阵（A 为你，B 为对手）：\nA=\n{A}\nB=\n{B}\n"
        "请判断一个可能的稳定选择（若存在纯策略纳什），并给出 (行动作, 列动作)。"
    ),
    # 表格/符号式（更简洁）
    (
        "[Game: {name}]\nRow actions = {acts_row}\nCol actions = {acts_col}\n"
        "A=\n{A}\nB=\n{B}\n"
        "Output a joint pure action from the sets above as '(Row, Col)'."
    ),
]

def _matrix_to_str(M: np.ndarray) -> str:
    return "\n".join(["  " + "  ".join([f"{v: .1f}" for v in row]) for row in M])

def _sample_prompt(game: Dict) -> str:
    A = game["A"]
    B = -A if game["B"] is None else game["B"]
    txt = random.choice(PROMPT_TEMPLATES)
    return txt.format(
        name=game["name"],
        acts_row=", ".join(game["actions_row"]),
        acts_col=", ".join(game["actions_col"]),
        A=_matrix_to_str(A),
        B=_matrix_to_str(B),
    )

class NashEnv(gym.Env):
    """
    一个用于 LLM 训练/评测的纳什均衡矩阵博弈环境（行玩家 vs 列玩家）。
    动作是一个二元组 (i, j) 表示选择的纯策略联动。
    奖励规则（默认）：若 (i,j) 是该博弈的纯策略纳什均衡之一，则 +1，否则 -1。
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 games: Optional[List[str]] = None,
                 seed: Optional[int] = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        # 可选子集；默认包含注册表的全部游戏
        self.game_keys = games if games is not None else list(GAMES_REGISTRY.keys())

        # 先初始化为占位，reset 时按具体游戏重设
        self.current_game_key: Optional[str] = None
        self.current_game: Optional[Dict] = None
        self.action_space: Optional[spaces.Space] = None
        self.observation_space = spaces.Dict({
            "description": spaces.Text(min_length=1, max_length=5000),
            "A": spaces.Box(low=-1e9, high=1e9, shape=(0, 0), dtype=np.float32),  # 动态
            "B": spaces.Box(low=-1e9, high=1e9, shape=(0, 0), dtype=np.float32),  # 动态
            "actions_row": spaces.Text(min_length=1, max_length=1024),
            "actions_col": spaces.Text(min_length=1, max_length=1024),
            "name": spaces.Text(min_length=1, max_length=256),
        })

    def _set_action_space(self, m: int, n: int):
        self.action_space = spaces.Tuple((spaces.Discrete(m), spaces.Discrete(n)))

    def _is_pure_nash(self, i: int, j: int) -> bool:
        assert self.current_game is not None
        return (i, j) in self.current_game["pure_ne"]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 随机抽一个博弈
        key = random.choice(self.game_keys)
        game = GAMES_REGISTRY[key].copy()
        if game["B"] is None:
            game["B"] = -game["A"]

        self.current_game_key = key
        self.current_game = game

        m, n = game["A"].shape
        self._set_action_space(m, n)

        description = _sample_prompt(game)

        obs = {
            "description": description,
            "A": game["A"].astype(np.float32),
            "B": game["B"].astype(np.float32),
            "actions_row": ", ".join(game["actions_row"]),
            "actions_col": ", ".join(game["actions_col"]),
            "name": game["name"],
        }
        info = {"game_key": key}
        return obs, info

    def step(self, action: Tuple[int, int]):
        assert self.current_game is not None, "Call reset() first."
        i, j = action
        m, n = self.current_game["A"].shape
        if not (0 <= i < m and 0 <= j < n):
            # 非法动作：强制 -1
            return None, -1.0, True, False, {"error": "invalid action index"}

        # 评分逻辑：仅对纯策略均衡给 +1；其余 -1
        # 对于只有混合均衡的游戏（如 MP/RPS），所有纯动作对都给 -1
        reward = 1.0 if self._is_pure_nash(i, j) else -1.0

        # 单步判分即可结束一个 episode
        terminated = True
        truncated = False
        obs = None
        info = {
            "game_key": self.current_game_key,
            "chosen_pair": (i, j),
            "is_pure_ne": (reward > 0),
            "has_mixed": self.current_game.get("has_mixed", False),
        }
        return obs, reward, terminated, truncated, info
