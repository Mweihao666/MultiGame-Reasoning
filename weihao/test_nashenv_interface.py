# weihao/test_nashenv_interface.py
from nashenv.env import NashEnv
from nashenv.config import NashEnvConfig

def test_interface_signature():
    env = NashEnv(NashEnvConfig())
    obs = env.reset(seed=0)
    assert isinstance(obs, str)

    obs2, reward, done, info = env.step(1)
    assert isinstance(obs2, str)
    assert isinstance(reward, int)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    assert "action_is_valid" in info
    assert "success" in info
    assert "NE" in info
