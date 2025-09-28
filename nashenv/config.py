# -*- coding: utf-8 -*-
"""
nashenv.config
- 提供默认的矩阵博弈注册表
- 后续需要可在这里加入难度采样、权重等配置
"""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import copy

def get_default_games() -> Dict[str, Dict[str, Any]]:
    """
    返回一组经典 2 人矩阵博弈的注册表。
    结构:
      {
        game_key: {
          "rows": [str, ...],
          "cols": [str, ...],
          "payoff": np.ndarray shape=(R, C, 2), dtype=float,
          "desc": str
        },
        ...
      }
    仅对“纯 NE”计分；无纯 NE 的博弈（如 MatchingPennies/RPS）默认 reward=0。
    """
    G: Dict[str, Dict[str, Any]] = {}

    def pack(rows: List[str], cols: List[str], pay, desc: str = ""):
        return {
            "rows": rows,
            "cols": cols,
            "payoff": np.array(pay, dtype=float),
            "desc": desc,
        }

    # Prisoner's Dilemma：NE = (D,D)
    G["PrisonersDilemma"] = pack(
        rows=["C", "D"],
        cols=["C", "D"],
        pay=[
            [(3, 3), (0, 5)],
            [(5, 0), (1, 1)],
        ],
        desc="Defect strictly dominates; (D,D) is the unique pure NE.",
    )

    # Coordination：NE = (A,A), (B,B)
    G["Coordination"] = pack(
        rows=["A", "B"],
        cols=["A", "B"],
        pay=[
            [(2, 2), (0, 0)],
            [(0, 0), (2, 2)],
        ],
        desc="Two pure NE on the diagonal.",
    )

    # Stag Hunt：NE = (Stag,Stag), (Hare,Hare)
    G["StagHunt"] = pack(
        rows=["Stag", "Hare"],
        cols=["Stag", "Hare"],
        pay=[
            [(3, 3), (0, 2)],
            [(2, 0), (2, 2)],
        ],
        desc="Payoff-dominant (Stag,Stag); risk-dominant (Hare,Hare).",
    )

    # Battle of the Sexes：NE = 两个对角
    G["BattleOfSexes"] = pack(
        rows=["Opera", "Football"],
        cols=["Opera", "Football"],
        pay=[
            [(2, 1), (0, 0)],
            [(0, 0), (1, 2)],
        ],
        desc="Two pure NE on the diagonal with asymmetric payoffs.",
    )

    # Matching Pennies（无纯 NE）
    G["MatchingPennies"] = pack(
        rows=["H", "T"],
        cols=["H", "T"],
        pay=[
            [(1, -1), (-1, 1)],
            [(-1, 1), (1, -1)],
        ],
        desc="Zero-sum; no pure NE.",
    )

    # Rock-Paper-Scissors（无纯 NE）
    rps = np.zeros((3, 3, 2), dtype=float)
    names = ["Rock", "Paper", "Scissors"]
    for i in range(3):
        for j in range(3):
            if i == j:
                rps[i, j, 0] = 0
                rps[i, j, 1] = 0
            elif (i - j) % 3 == 1:  # row wins
                rps[i, j, 0] = 1
                rps[i, j, 1] = -1
            else:
                rps[i, j, 0] = -1
                rps[i, j, 1] = 1
    G["RockPaperScissors"] = {
        "rows": names,
        "cols": names,
        "payoff": rps,
        "desc": "Zero-sum; no pure NE.",
    }

    # ZeroSum2x2（一般也无纯 NE；用于覆盖面）
    G["ZeroSum2x2"] = pack(
        rows=["R0", "R1"],
        cols=["C0", "C1"],
        pay=[
            [(1, -1), (0, 0)],
            [(0, 0), (1, -1)],
        ],
        desc="Illustrative zero-sum 2x2 (likely no pure NE).",
    )

    return copy.deepcopy(G)
