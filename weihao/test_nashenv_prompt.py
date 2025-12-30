# weihao/test_nashenv_prompt.py
from nashenv.env import NashEnv
from nashenv.config import NashEnvConfig

def test_prompt_deterministic():
    cfg = NashEnvConfig(game="PD")
    env = NashEnv(cfg)
    obs1 = env.reset(seed=123)
    obs2 = env.reset(seed=123)
    assert obs1 == obs2  # 同一seed输出一致

def test_prompt_contains_payoff():
    cfg = NashEnvConfig(game="PD")
    env = NashEnv(cfg)
    obs = env.reset(seed=1)
    # payoff矩阵的数字一定要出现在prompt里
    for val in [3, 0, 5, 1]:
        assert str(val) in obs
