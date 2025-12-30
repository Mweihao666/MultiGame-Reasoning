import importlib

def main():
    MODULE = "nashenv.env"
    CLASS  = "NashEnv"
    EnvCls = getattr(importlib.import_module(MODULE), CLASS)

    # 创建环境
    env = EnvCls()

    # reset 测试：只返回 str
    obs = env.reset(seed=42)
    print("reset obs:", obs)
    assert isinstance(obs, str)

    # render 测试：同状态两次一致
    r1, r2 = env.render(), env.render()
    print("render outputs:", r1, r2)
    assert r1 == r2

    # 获取所有动作
    actions = env.get_all_actions()
    print("valid actions:", actions)
    assert actions == [1, 2], "动作索引必须从 1 开始"

    # step 测试：返回四元组
    obs, reward, done, info = env.step(actions[0])
    print("step return:", obs, reward, done, info)

    assert isinstance(obs, str)
    assert reward in {0, 1}
    assert done is True
    for key in ["action_is_valid", "action_is_effective", "success"]:
        assert key in info

    env.close()
    print("✅ NashEnv 接口验证通过！")

if __name__ == "__main__":
    main()
