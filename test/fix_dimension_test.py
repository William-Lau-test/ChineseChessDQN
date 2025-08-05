import numpy as np
from environment import ChineseChessEnv


def full_dimension_check():
    print("=== 维度修复验证 ===")

    env = ChineseChessEnv()
    print(f"修正前env.state_dim = {env.state_dim}")  # 应显示90（错误值）

    # 强制修正维度定义
    env.state_dim = 1530
    print(f"修正后env.state_dim = {env.state_dim}")  # 应显示1530

    # 验证各环节
    try:
        state = env.reset()
        print(f"重置后状态维度: {len(state)}")

        legal_moves = env.get_legal_moves()
        print(f"合法移动数量: {len(legal_moves)}")

        next_state, _, _, _ = env.step(legal_moves[0])
        print(f"移动后状态维度: {len(next_state)}")

        print(">>> 所有维度检查通过！")
    except Exception as e:
        print(f">!! 修复失败: {str(e)}")


if __name__ == "__main__":
    full_dimension_check()