# weihao/test_nashenv_nashpy.py
import pytest
from nashenv.env import NashEnv
from nashenv.config import NashEnvConfig

@pytest.mark.parametrize("game,expected_ne", [
    ("PD", [(1, 1)]),                  # 囚徒困境: 双方都选背叛
    ("SH", [(0, 0), (1, 1)]),          # 协调博弈(Stag Hunt): 两个均衡
    ("MP", []),                        # 匹配硬币(Matching Pennies): 没有纯 NE
])
def test_pure_nash_equilibria(game, expected_ne):
    cfg = NashEnvConfig(game=game)
    env = NashEnv(cfg)
    env.reset(seed=0)
    ne = env._pure_nash_equilibria()
    assert sorted(ne) == sorted(expected_ne)
