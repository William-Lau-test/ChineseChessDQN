import numpy as np
from environment import ChineseChessEnv


def analyze_state():
    env = ChineseChessEnv()
    state = env.reset()

    print("=== 状态分析报告 ===")
    print(f"理论维度: {env.state_dim}")
    print(f"实际维度: {len(state)}")

    # 分析各通道
    channels = {
        "基础棋盘层": state[0:90],
        "红方车": state[90:180],
        "黑方车": state[180:270],
        # 添加其他通道分析...
        "当前控制区": state[1350:1440],
        "对手控制区": state[1440:1530]
    }

    for name, channel in channels.items():
        print(f"\n{name} (min/max): {np.min(channel):.1f}/{np.max(channel):.1f}")
        print(f"非零值数量: {np.count_nonzero(channel)}")


if __name__ == "__main__":
    analyze_state()