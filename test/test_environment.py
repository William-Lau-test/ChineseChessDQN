import numpy as np
from environment import ChineseChessEnv


def test_environment():
    env = ChineseChessEnv()

    # 测试状态维度
    state = env.reset()
    print(f"状态类型: {type(state)}")  # 应为numpy.ndarray
    print(f"状态形状: {state.shape}")  # 应为(1530,)

    # 测试合法移动
    legal_moves = env.get_legal_moves()
    print(f"初始合法移动数: {len(legal_moves)}")  # 红方应有约40个合法移动

    # 测试状态变化
    sample_move = legal_moves[0]
    next_state, _, _, _ = env.step(sample_move)
    print(f"移动后状态变化: {np.sum(state != next_state)}")  # 应有变化


if __name__ == "__main__":
    test_environment()