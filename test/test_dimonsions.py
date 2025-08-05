import numpy as np
from environment import ChineseChessEnv
from dqn import ChessDQN


def verify_dimensions():
    env = ChineseChessEnv()
    state = env.reset()

    print("[验证阶段1] 环境声明 vs 实际状态")
    print(f"env.state_dim = {env.state_dim} | 实际shape = {state.shape}")
    assert env.state_dim == state.shape[0], "维度声明不一致！"

    print("\n[验证阶段2] 网络输入验证")
    model = ChessDQN(env.state_dim)
    try:
        output = model(state)
        print(f"网络测试通过！输出形状: {output.shape}")
    except Exception as e:
        print(f"网络测试失败: {str(e)}")
        print("可能的原因:")
        print("1. environment.py中的state_dim未更新")
        print("2. get_state()返回的维度不正确")
        print("3. 网络初始化参数未传递正确")


if __name__ == "__main__":
    verify_dimensions()