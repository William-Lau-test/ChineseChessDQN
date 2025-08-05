import time
import pygame
from dqn import DQNAgent
from environment import ChineseChessEnv


def ai_vs_ai_battle(difficulty='medium', render=True, move_delay=0.5):
    """两个AI对战"""
    env = ChineseChessEnv()
    red_ai = DQNAgent(env, difficulty)
    black_ai = DQNAgent(env, difficulty)

    state = env.reset()
    while True:
        if render:
            env.render_pygame()
            time.sleep(move_delay)

        current_ai = red_ai if env.current_player == 1 else black_ai
        legal_moves = env.get_legal_moves()
        action = current_ai.select_action(state, legal_moves) if legal_moves else None

        if action is None:
            print("No legal moves! Game over.")
            break

        state, _, done, _ = env.step(action)
        if done:
            winner = "Red" if env.current_player == -1 else "Black"
            print(f"{winner} wins!")
            break


def human_vs_ai(human_side='red', ai_difficulty='medium'):
    env = ChineseChessEnv()
    agent = DQNAgent(env, ai_difficulty)

    episode = 0
    state = env.reset()

    while True:
        env.render_pygame()

        if env.current_player == (1 if human_side == 'red' else -1):
            # 人类回合
            action = env.get_move_from_clicks()
        else:
            # AI回合
            legal_moves = env.get_legal_moves()
            action = agent.select_action(state, legal_moves, episode)
            time.sleep(0.5)
            episode += 1

        if action is None:
            break

        state, _, done, _ = env.step(action)
        if done:
            print("Game Over!")
            break