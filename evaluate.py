from datetime import time
from typing import Tuple
from dqn import DQNAgent
from environment import ChineseChessEnv


def evaluate(agent, env, episodes=20, render=False):
    """评估函数
    参数:
        agent: DQNAgent实例
        env: 象棋环境
        episodes: 评估回合数 (默认20)
        render: 是否渲染 (默认False)
    """
    win_count = 0
    total_reward = 0

    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            legal_moves = env.get_legal_moves()
            if not legal_moves:
                break

            action = agent.select_action(state, legal_moves, epsilon=0.05)
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        total_reward += episode_reward
        if reward > 0:  # 假设正奖励表示胜利
            win_count += 1

    win_rate = win_count / episodes
    avg_reward = total_reward / episodes
    return win_rate, avg_reward