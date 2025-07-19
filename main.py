import argparse
import numpy as np
import matplotlib.pyplot as plt
import pygame
from environment import ChineseChessEnv
from dqn import DQN
from utils import select_action
import logging

# 配置日志
logging.basicConfig(filename='training.log', level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
verbose = False

def log_info(message, verbose_only=False):
    if verbose or not verbose_only:
        logger.info(message)

def log_warning(message):
    logger.warning(message)

def log_error(message):
    logger.error(message)

EPISODES = 2000
MAX_STEPS = 200
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 5000
RENDER_GUI = False
BATCH_SIZE = 128
WARMUP_STEPS = 1000

def parse_args():
    parser = argparse.ArgumentParser(description="DQN for Chinese Chess")
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], default='hard')
    parser.add_argument('--render-gui', action='store_true', default=RENDER_GUI,
                        help='Render GUI during training')
    parser.add_argument('--ai-vs-ai', action='store_true', help='Enable AI vs AI mode')
    return parser.parse_args()

def get_player_side():
    while True:
        side = input("Choose your side (red/black): ").strip().lower()
        if side in ['red', 'black']:
            return 1 if side == 'red' else -1
        print("Invalid choice. Please enter 'red' or 'black'.")

def ai_vs_ai(agent, render_gui=False, episodes=5):
    env = ChineseChessEnv(computer_player=1) #, human_player=-1
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = select_action(agent, state, epsilon=0.0, env=env)
            if action is None:
                break
            state, reward, done = env.step(action)
            ep_reward += reward
            if render_gui:
                env.render_pygame()
                pygame.time.wait(300)
        print(f"AI vs AI Episode {ep+1}, Reward: {ep_reward:.2f}")
        rewards.append(ep_reward)
    print(f"AI vs AI Average Reward: {np.mean(rewards):.2f}")
    env.close()

def train(computer_player=1, render_gui=RENDER_GUI):
    env = ChineseChessEnv(computer_player=computer_player)
    state_dim = env.state_dim
    action_dim = len(env.get_legal_moves())
    agent = DQN(state_dim, action_dim, batch_size=BATCH_SIZE)
    rewards, avg_losses, invalid_actions = [], [], []
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        state = env.reset()
        ep_reward, ep_losses, ep_invalid = 0, [], 0
        if render_gui:
            env.render_pygame()
        for t in range(MAX_STEPS):
            action = select_action(agent, state, epsilon, env)
            if action is None:
                break
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done, env)
            agent.update(env)
            state = next_state
            ep_reward += reward
            if reward <= -1.0:
                ep_invalid += 1
            if agent.losses:
                ep_losses.append(agent.losses[-1])
            if render_gui:
                env.render_pygame()
                pygame.time.wait(300)
            if done:
                break
        rewards.append(ep_reward)
        avg_losses.append(np.mean(ep_losses) if ep_losses else 0)
        invalid_actions.append(ep_invalid)
        epsilon = max(EPSILON_END, EPSILON_START * np.exp(-episode / EPSILON_DECAY))
    env.close()
    return agent

def test(agent, env, episodes=5):
    test_rewards = []
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        env.render_pygame()
        for t in range(MAX_STEPS):
            if env.current_player == env.computer_player:
                action = select_action(agent, state, epsilon=0.0, env=env)
            else:
                action = env.get_move_from_clicks()
                if action is None:
                    continue
            next_state, reward, done = env.step(action)
            state = next_state
            episode_reward += reward
            env.render_pygame()
            if done:
                pygame.time.wait(2000)
                break
        test_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}, Reward: {episode_reward:.2f}")
    print(f"Test Mean Reward: {np.mean(test_rewards):.2f}, Std: {np.std(test_rewards):.2f}")
    env.close()

if __name__ == "__main__":
    args = parse_args()
    render_gui = args.render_gui

    if args.ai_vs_ai:
        env = ChineseChessEnv(computer_player=1) #, human_player=-1
        agent = DQN(env.state_dim, len(env.get_legal_moves()))
        try:
            agent.load(f'dqn_{args.difficulty}.pth')
        except:
            print(f"Model not found. Training from scratch.")
            agent = train(computer_player=1, render_gui=render_gui)
        ai_vs_ai(agent, render_gui=render_gui)
    else:
        human_player = get_player_side()
        computer_player = -human_player
        env = ChineseChessEnv(computer_player=computer_player)
        agent = DQN(env.state_dim, len(env.get_legal_moves()))
        try:
            agent.load(f'dqn_{args.difficulty}.pth')
        except:
            print(f"Model not found. Training from scratch.")
            agent = train(computer_player=computer_player, render_gui=render_gui)
        test_env = ChineseChessEnv(computer_player=computer_player, agent=agent)
        test(agent, test_env)
