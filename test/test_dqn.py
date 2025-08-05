import torch
import numpy as np
from environment import ChineseChessEnv
from dqn import ChessDQN


def test_dimensions():
    env = ChineseChessEnv()
    state = env.reset()

    print(f"环境状态维度: {env.state_dim}")  # 应为1530
    print(f"实际状态形状: {state.shape}")  # 应为(1530,)

    # 测试网络
    model = ChessDQN(env.state_dim)

    # 测试单样本
    input_tensor = torch.FloatTensor(state).unsqueeze(0)
    print(f"网络输入形状: {input_tensor.shape}")  # 应为[1, 1530]

    output = model(input_tensor)
    print(f"网络输出形状: {output.shape}")  # 应为[1, 8100]

    # 验证反向传播
    dummy_loss = output.mean()
    dummy_loss.backward()
    print("反向传播测试通过")


if __name__ == "__main__":
    test_dimensions()