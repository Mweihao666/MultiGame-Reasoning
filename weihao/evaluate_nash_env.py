from __future__ import annotations
import sys, re, os
from typing import Tuple, Dict, Any
import numpy as np
from nashenv.env import NashEnv   # 导入新环境

# ---------------- 适配器 ----------------
def adapt_obs(obs: Dict[str,Any]) -> Dict[str,Any]:
    """
    把 NashEnv 的 obs 转换成旧版 evaluate 逻辑需要的格式
    """
    return {
        "description": obs.get("text", ""),
        "actions_row": ",".join(obs["rows"]),
        "actions_col": ",".join(obs["cols"]),
        # 只取行玩家收益矩阵 A (R,C)，方便 random_policy 用
        "A": np.array(obs["payoff_matrix"])[:,:,0],
        "game_key": obs["game_key"],
    }

# ---------------- 手工解析命令行参数 ----------------
def parse_cli(argv):
    episodes = 200
    policy = "random"
    mode = "row"   # 新增：默认 row
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--episodes":
            episodes = int(argv[i+1]); i += 2; continue
        if a.startswith("--episodes="):
            episodes = int(a.split("=",1)[1]); i += 1; continue
        if a == "--policy":
            policy = argv[i+1]; i += 2; continue
        if a.startswith("--policy="):
            policy = a.split("=",1)[1]; i += 1; continue
        i += 1
    return episodes, policy, mode

# ---------------- 解析器 & LLM 策略 ----------------
def parse_joint_action(text: str, actions_row, actions_col):
    m = re.search(r"\(\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*\)", text)
    if m:
        rname, cname = m.group(1), m.group(2)
        if rname in actions_row and cname in actions_col:
            return actions_row.index(rname), actions_col.index(cname)
    m2 = re.search(r"Row\s*=\s*([A-Za-z]+).*?Col(?:umn)?\s*=\s*([A-Za-z]+)", text, flags=re.I|re.S)
    if m2:
        rname, cname = m2.group(1), m2.group(2)
        if rname in actions_row and cname in actions_col:
            return actions_row.index(rname), actions_col.index(cname)
    cand_r = [a for a in actions_row if re.search(rf"\b{re.escape(a)}\b", text, flags=re.I)]
    cand_c = [a for a in actions_col if re.search(rf"\b{re.escape(a)}\b", text, flags=re.I)]
    if len(cand_r) == 1 and len(cand_c) == 1:
        return actions_row.index(cand_r[0]), actions_col.index(cand_c[0])
    raise ValueError(f"无法解析动作: {text}")

def llm_policy(obs, model_call_fn):
    prompt = obs["description"]
    output = model_call_fn(prompt)
    actions_row = [a.strip() for a in obs["actions_row"].split(",")]
    actions_col = [a.strip() for a in obs["actions_col"].split(",")]
    return parse_joint_action(output, actions_row, actions_col)

# ---------------- OpenAI API 接入 ----------------
try:
    from openai import OpenAI
    _OPENAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    def openai_model(prompt: str) -> str:
        prompt2 = prompt + "\n\nReturn exactly one line like: (RowAction, ColAction). No extra words."
        resp = _OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt2}],
            temperature=0
        )
        return resp.choices[0].message.content or ""
except Exception:
    openai_model = None

# ---------------- 随机策略 ----------------
def random_policy(obs: Dict[str,Any]) -> Tuple[int,int]:
    A = obs["A"]; m, n = A.shape
    return np.random.randint(0,m), np.random.randint(0,n)

# ---------------- 评测循环 ----------------
def evaluate(env: NashEnv, policy_fn, episodes: int = 200) -> Dict[str,float]:
    total, correct = 0, 0
    per_game = {}
    for _ in range(episodes):
        raw_obs, info = env.reset()
        obs = adapt_obs(raw_obs)
        try:
            if env.eval_mode == "row":
                i, _ = policy_fn(obs)       # 只用行动作
                action = i
            else:
                i, j = policy_fn(obs)
                action = (i, j)
        except Exception:
            action = 0 if env.eval_mode == "row" else (0, 0)
        _, reward, _, info2 = env.step(action)
        total += 1
        gk = info["game_key"]
        per_game.setdefault(gk, {"n":0,"acc":0.0})
        per_game[gk]["n"] += 1
        if reward > 0:
            correct += 1
            per_game[gk]["acc"] += 1.0
    overall = correct/total if total else 0.0
    for gk, rec in per_game.items():
        rec["acc"] = rec["acc"]/rec["n"] if rec["n"] else 0.0
    return {"overall_acc": overall, "per_game": per_game}


# ---------------- 主函数 ----------------
def main():
    episodes, policy, mode = parse_cli(sys.argv[1:])
    env = NashEnv(seed=123, eval_mode=mode)  # 评测用 joint 模式

    if policy == "random":
        policy_fn = random_policy
    else:
        if openai_model is None:
            def dummy_model(prompt: str) -> str:
                if "Stag Hunt" in prompt:
                    return "(Stag, Stag)"
                return "(Cooperate, Cooperate)"
            policy_fn = lambda obs: llm_policy(obs, dummy_model)
        else:
            policy_fn = lambda obs: llm_policy(obs, openai_model)

    stats = evaluate(env, policy_fn, episodes=episodes)
    print(f"Policy: {policy}  Episodes: {episodes}")
    print("Overall accuracy:", round(stats["overall_acc"], 3))
    for gk, rec in sorted(stats["per_game"].items()):
        print(f"{gk:>20s}  n={rec['n']:4d}  acc={rec['acc']:.3f}")

if __name__ == "__main__":
    main()
