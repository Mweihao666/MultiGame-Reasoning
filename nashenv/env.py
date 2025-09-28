# -*- coding: utf-8 -*-
"""
nashenv.env
- NashEnv: 与 bandit 风格对齐的矩阵博弈环境
- 仅纯 NE 计分（reward∈{0,1}）
- eval_mode in {"row","joint"}，默认 "row"
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
import random
import json

from .config import get_default_games

ObsType = Dict[str, Any]
ActionType = Union[int, Tuple[int, int]]

class NashEnv:
    def __init__(
        self,
        games: Optional[Dict[str, Dict[str, Any]]] = None,
        *,
        eval_mode: str = "row",
        seed: Optional[int] = None,
        include_text_prompt: bool = True,
    ) -> None:
        assert eval_mode in {"row", "joint"}
        self.eval_mode = eval_mode
        self.include_text_prompt = include_text_prompt

        # 可复现随机
        self._seed = seed if seed is not None else np.random.SeedSequence().entropy
        self.np_rng = np.random.default_rng(self._seed)
        self.py_rng = random.Random(int(self._seed))

        # 游戏注册表
        self.games = games if games is not None else get_default_games()

        # 状态
        self._cur_key: Optional[str] = None
        self._cur_game: Optional[Dict[str, Any]] = None
        self._cur_obs: Optional[ObsType] = None
        self._done: bool = True

    # ---------- API ----------
    def reset(self) -> Tuple[ObsType, Dict[str, Any]]:
        self._cur_key, self._cur_game = self._sample_game()
        rows: List[str] = self._cur_game["rows"]
        cols: List[str] = self._cur_game["cols"]
        payoff = self._cur_game["payoff"]  # (R,C,2)

        pure_ne = self._compute_pure_ne(payoff)
        pure_ne_rows = sorted({r for (r, _) in pure_ne})

        obs: ObsType = {
            "game_key": self._cur_key,
            "rows": rows,
            "cols": cols,
            "payoff_matrix": payoff.tolist(),
            "pure_ne_rows_hint": None,  # 训练不暴露；占位
        }
        if self.include_text_prompt:
            obs["text"] = self._render_prompt(self._cur_key, rows, cols, payoff, mode=self.eval_mode)

        self._cur_obs = obs
        self._done = False

        info = {
            "game_key": self._cur_key,
            "eval_mode": self.eval_mode,
            "seed": int(self._seed),
        }
        return obs, info

    def step(self, action: ActionType) -> Tuple[ObsType, int, bool, Dict[str, Any]]:
        assert not self._done, "Episode has finished. Call reset() before step()."
        assert self._cur_game is not None and self._cur_obs is not None

        rows: List[str] = self._cur_game["rows"]
        cols: List[str] = self._cur_game["cols"]
        payoff = np.array(self._cur_game["payoff"], dtype=float)
        R, C, _ = payoff.shape

        parsed = self._parse_action(action, R, C)
        row_idx: Optional[int] = parsed.get("row_idx")
        col_idx: Optional[int] = parsed.get("col_idx")

        pure_ne = self._compute_pure_ne(payoff)
        pure_ne_rows = sorted({r for (r, _) in pure_ne})

        is_correct = False
        reason = ""
        if len(pure_ne) == 0:
            is_correct = False
            reason = "No pure NE exists (mixed-only game)."
        else:
            if self.eval_mode == "row":
                if row_idx is None or not (0 <= row_idx < R):
                    is_correct = False
                    reason = f"Invalid row action: {row_idx}."
                else:
                    is_correct = row_idx in pure_ne_rows
                    if is_correct:
                        matched_cols = sorted([c for (r, c) in pure_ne if r == row_idx])
                        reason = f"Row {row_idx} is part of pure NE rows; NE columns: {matched_cols}."
                    else:
                        reason = f"Row {row_idx} is not part of any pure NE row."
            else:
                if (row_idx is None or not (0 <= row_idx < R) or
                        col_idx is None or not (0 <= col_idx < C)):
                    is_correct = False
                    reason = f"Invalid joint action: ({row_idx}, {col_idx})."
                else:
                    is_correct = (row_idx, col_idx) in pure_ne
                    reason = f"({row_idx}, {col_idx}) is {'a' if is_correct else 'not a'} pure NE."

        reward: int = 1 if is_correct else 0
        self._done = True

        info: Dict[str, Any] = {
            "game_key": self._cur_key,
            "eval_mode": self.eval_mode,
            "action_raw": action,
            "action_parsed": parsed,
            "is_correct": bool(is_correct),
            "reward": int(reward),
            "pure_ne_list": pure_ne,
            "pure_ne_rows": pure_ne_rows,
            "reason": reason,
        }
        return self._cur_obs, reward, True, info

    # ---------- utils ----------
    def seed(self, seed: int) -> None:
        self._seed = int(seed)
        self.np_rng = np.random.default_rng(self._seed)
        self.py_rng = random.Random(self._seed)

    def _sample_game(self) -> Tuple[str, Dict[str, Any]]:
        keys = list(self.games.keys())
        key = self.py_rng.choice(keys)
        return key, self.games[key]

    @staticmethod
    def _compute_pure_ne(payoff: np.ndarray) -> List[Tuple[int, int]]:
        R, C, _ = payoff.shape
        row_best_mask = np.zeros((R, C), dtype=bool)
        for c in range(C):
            col_slice = payoff[:, c, 0]
            max_val = np.max(col_slice)
            row_best_mask[:, c] = np.isclose(col_slice, max_val)

        col_best_mask = np.zeros((R, C), dtype=bool)
        for r in range(R):
            row_slice = payoff[r, :, 1]
            max_val = np.max(row_slice)
            col_best_mask[r, :] = np.isclose(row_slice, max_val)

        ne_mask = row_best_mask & col_best_mask
        return [(r, c) for r in range(R) for c in range(C) if ne_mask[r, c]]

    def _render_prompt(
        self,
        game_key: str,
        rows: List[str],
        cols: List[str],
        payoff: np.ndarray,
        mode: str = "row",
    ) -> str:
        header = f"[Game: {game_key}]"
        instr = (
            "You are the ROW player. "
            "Return the INDEX of your chosen action (an integer).\n"
            "Scoring rule: only PURE Nash equilibria are counted as correct.\n"
        )
        if mode == "row":
            instr += "Evaluation mode: ROW-ONLY — pick a row index that belongs to at least one pure NE.\n"
        else:
            instr += "Evaluation mode: JOINT — pick (row, col) or a row-major linear index.\n"

        r_names = ", ".join([f"{i}:{name}" for i, name in enumerate(rows)])
        c_names = ", ".join([f"{j}:{name}" for j, name in enumerate(cols)])

        table_lines = ["Payoff matrix [row, col] -> (u_row, u_col):"]
        for i in range(payoff.shape[0]):
            cells = []
            for j in range(payoff.shape[1]):
                u_r = payoff[i, j, 0]
                u_c = payoff[i, j, 1]
                cells.append(f"({u_r:.2f},{u_c:.2f})")
            table_lines.append(f"Row {i}: " + " | ".join(cells))

        return "\n".join(
            [header, instr, f"Row actions: {r_names}", f"Col actions: {c_names}", "", *table_lines]
        )

    def _parse_action(self, action: ActionType, R: int, C: int) -> Dict[str, Optional[int]]:
        result: Dict[str, Optional[int]] = {"row_idx": None, "col_idx": None, "decoded_from": None}
        if self.eval_mode == "row":
            if isinstance(action, int):
                result["row_idx"] = action
                result["decoded_from"] = "row-only:int"
            else:
                result["decoded_from"] = f"row-only:invalid({type(action).__name__})"
            return result

        if isinstance(action, (tuple, list)) and len(action) == 2:
            r, c = int(action[0]), int(action[1])
            result["row_idx"] = r
            result["col_idx"] = c
            result["decoded_from"] = "tuple"
        elif isinstance(action, int):
            r = action // C
            c = action % C
            result["row_idx"] = r
            result["col_idx"] = c
            result["decoded_from"] = "int(row-major)"
        else:
            result["decoded_from"] = f"invalid({type(action).__name__})"
        return result


# --------- 快速本地自检（可选） ---------
if __name__ == "__main__":
    env = NashEnv(eval_mode="row", seed=42, include_text_prompt=False)
    obs, info = env.reset()
    print("[RESET]", info)
    obs2, reward, done, info2 = env.step(1)
    print("[STEP]", reward, done, json.dumps(info2, indent=2))
