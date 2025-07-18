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
    return parser.parse_args()

def get_player_side():
    while True:
        side = input("Choose your side (red/black): ").strip().lower()
        if side in ['red', 'black']:
            return 1 if side == 'red' else -1
        print("Invalid choice. Please enter 'red' or 'black'.")

def train(computer_player=1, render_gui=RENDER_GUI):
    env = ChineseChessEnv(computer_player=computer_player)
    state_dim = env.state_dim
    action_dim = len(env.get_legal_moves())  # 动态获取初始合法动作数量
    agent = DQN(state_dim, action_dim, batch_size=BATCH_SIZE)

    rewards = []
    avg_losses = []
    invalid_actions = []
    epsilon = EPSILON_START

    log_info(f"Starting training with computer_player={computer_player}, render_gui={render_gui}")

    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        episode_invalid = 0
        if render_gui:
            env.render_pygame()
        for t in range(MAX_STEPS):
            try:
                action = select_action(agent, state, epsilon, env)
                if action is None:
                    log_warning(f"Episode {episode + 1}, step {t + 1}: No legal moves, terminating")
                    reward = -1 if env.current_player == env.computer_player else 1
                    episode_reward += reward
                    if render_gui:
                        env.render_pygame()
                    break
                log_info(f"Episode {episode + 1}, Step {t + 1}: Selected action={action}", verbose_only=True)

                next_state, reward, done = env.step(action)
                log_info(f"Episode {episode + 1}, Step {t + 1}: Received reward={reward}", verbose_only=True)
                agent.store_transition(state, action, reward, next_state, done, env)
                agent.update(env)

                state = next_state
                episode_reward += reward
                if reward <= -1.0:
                    episode_invalid += 1
                if agent.losses:
                    episode_losses.append(agent.losses[-1])
                if render_gui:
                    env.render_pygame()
                    pygame.time.wait(500)
                if done:
                    log_info(f"Episode {episode + 1} ended: Reward={episode_reward:.2f}, Invalid Actions={episode_invalid}")
                    break
            except Exception as e:
                log_error(f"Error in episode {episode + 1}, step {t + 1}: {e}")
                raise

        rewards.append(episode_reward)
        avg_losses.append(np.mean(episode_losses) if episode_losses else 0)
        invalid_actions.append(episode_invalid)
        epsilon = max(EPSILON_END, EPSILON_START * np.exp(-episode / EPSILON_DECAY))
        log_info(f"Episode {episode + 1}/{EPISODES}, Reward: {episode_reward:.2f}, "
                 f"Avg Loss: {avg_losses[-1]:.4f}, Invalid Actions: {episode_invalid}, "
                 f"Epsilon: {epsilon:.3f}")
        if episode % 100 == 0:
            print(f"Episode {episode + 1}/{EPISODES}, Reward: {episode_reward:.2f}, "
                  f"Avg Loss: {avg_losses[-1]:.4f}, Invalid Actions: {episode_invalid}, "
                  f"Epsilon: {epsilon:.3f}")

        if len(rewards) >= 100:
            last_100_rewards = rewards[-100:]
            print(f"Last 100 episodes - Mean Reward: {np.mean(last_100_rewards):.2f}, "
                  f"Std: {np.std(last_100_rewards):.2f}, "
                  f"Mean Invalid Actions: {np.mean(invalid_actions[-100:]):.2f}")
            log_info(f"Last 100 episodes - Mean Reward: {np.mean(last_100_rewards):.2f}, "
                     f"Std: {np.std(last_100_rewards):.2f}, "
                     f"Mean Invalid Actions: {np.mean(invalid_actions[-100:]):.2f}")

        if episode + 1 in [EPISODES/10, EPISODES/5, EPISODES/2, EPISODES]:
            try:
                agent.save(f'dqn_{["easy", "medium", "hard", "expert"][[EPISODES/10, EPISODES/5, EPISODES/2, EPISODES].index(episode + 1)]}.pth')
                print(f"Saved {['easy', 'medium', 'hard', 'expert'][[EPISODES/10, EPISODES/5, EPISODES/2, EPISODES].index(episode + 1)]} model")
                log_info(f"Saved {['easy', 'medium', 'hard', 'expert'][[EPISODES/10, EPISODES/5, EPISODES/2, EPISODES].index(episode + 1)]} model")
            except Exception as e:
                log_error(f"Failed to save model at episode {episode + 1}: {e}")

        if episode % 10 == 0:
            try:
                if rewards:
                    def smooth(y, box_pts=10):
                        box = np.ones(box_pts)/box_pts
                        return np.convolve(y, box, mode='same')

                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 3, 1)
                    plt.plot(smooth(rewards), label='Smoothed')
                    plt.xlabel('Episode')
                    plt.ylabel('Reward')
                    plt.title('Training Reward Curve')
                    plt.subplot(1, 3, 2)
                    plt.plot(avg_losses)
                    plt.xlabel('Episode')
                    plt.ylabel('Loss')
                    plt.title('Training Loss Curve')
                    plt.subplot(1, 3, 3)
                    plt.plot(invalid_actions)
                    plt.xlabel('Episode')
                    plt.ylabel('Invalid Actions')
                    plt.title('Invalid Actions per Episode')
                    plt.tight_layout()
                    plt.savefig('training_curves.png')
                    plt.close()
                    log_info(f"Saved training_curves.png at episode {episode + 1}")
            except Exception as e:
                log_error(f"Error saving plot at episode {episode + 1}: {e}")

    try:
        if rewards:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 3, 1)
            plt.plot(rewards, color='#1f77b4')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Final Training Reward Curve')
            plt.subplot(1, 3, 2)
            plt.plot(avg_losses, color='#ff7f0e')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.title('Final Training Loss Curve')
            plt.subplot(1, 3, 3)
            plt.plot(invalid_actions, color='#2ca02c')
            plt.xlabel('Episode')
            plt.ylabel('Invalid Actions')
            plt.title('Final Invalid Actions per Episode')
            plt.tight_layout()
            plt.savefig('final_training_curves.png')
            plt.close()
            log_info("Saved final_training_curves.png")
    except Exception as e:
        log_error(f"Error saving final plot: {e}")

    env.close()
    log_info("Training completed")
    return agent

def test(agent, env, episodes=5):
    test_rewards = []
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        env.render_pygame()
        print(f"\nTest Episode {episode + 1}")
        log_info(f"Starting test episode {episode + 1}")
        for t in range(MAX_STEPS):
            try:
                if env.current_player == env.computer_player:
                    action = select_action(agent, state, epsilon=0.0, env=env)
                    if action is None:
                        print(f"Test episode {episode + 1}, step {t + 1}: No legal moves for computer")
                        log_warning(f"Test episode {episode + 1}, step {t + 1}: No legal moves for computer")
                        reward = -1
                        episode_reward += reward
                        break
                    print(f"Computer ({'Red' if env.current_player == 1 else 'Black'}) moves: {action}")
                    log_info(f"Computer ({'Red' if env.current_player == 1 else 'Black'}) moves: {action}")
                else:
                    legal_moves = env.get_legal_moves()
                    if not legal_moves:
                        print(f"Test episode {episode + 1}, step {t + 1}: No legal moves for human")
                        log_warning(f"Test episode {episode + 1}, step {t + 1}: No legal moves for human")
                        reward = 1 if env.current_player == env.computer_player else -1
                        episode_reward += reward
                        break
                    print(f"Your turn ({'Red' if env.current_player == 1 else 'Black'}): Click to select piece and destination")
                    action = env.get_move_from_clicks()
                    if action is None:
                        print(f"Test episode {episode + 1}, step {t + 1}: Invalid human move, skipping")
                        log_warning(f"Test episode {episode + 1}, step {t + 1}: Invalid human move, skipping")
                        continue
                    print(f"Human ({'Red' if env.current_player == 1 else 'Black'}) moves: {action}")
                    log_info(f"Human ({'Red' if env.current_player == 1 else 'Black'}) moves: {action}")

                next_state, reward, done = env.step(action)
                state = next_state
                episode_reward += reward
                env.render_pygame()
                if done:
                    print(f"Game Over! Reward: {reward:.2f}")
                    log_info(f"Game Over! Reward: {reward:.2f}")
                    env.render_pygame()
                    pygame.time.wait(2000)
                    break
            except Exception as e:
                print(f"Error in test episode {episode + 1}, step {t + 1}: {e}")
                log_error(f"Error in test episode {episode + 1}, step {t + 1}: {e}")
                raise
        test_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}, Reward: {episode_reward:.2f}")
        log_info(f"Test Episode {episode + 1}, Reward: {episode_reward:.2f}")
    print(f"Test Mean Reward: {np.mean(test_rewards):.2f}, Std: {np.std(test_rewards):.2f}")
    log_info(f"Test Mean Reward: {np.mean(test_rewards):.2f}, Std: {np.std(test_rewards):.2f}")
    env.close()

if __name__ == "__main__":
    args = parse_args()
    render_gui = args.render_gui

    human_player = get_player_side()
    computer_player = -human_player
    print(f"Human plays as {'Red' if human_player == 1 else 'Black'}, Computer plays as {'Red' if computer_player == 1 else 'Black'}")
    log_info(f"Human plays as {'Red' if human_player == 1 else 'Black'}, Computer plays as {'Red' if computer_player == 1 else 'Black'}")

    env = ChineseChessEnv(computer_player=computer_player)
    action_dim = len(env.get_legal_moves())  # 动态获取初始合法动作数量
    agent = DQN(env.state_dim, action_dim)

    try:
        if args.difficulty == 'easy':
            agent.load('dqn_easy.pth')
        elif args.difficulty == 'medium':
            agent.load('dqn_medium.pth')
        elif args.difficulty == 'hard':
            agent.load('dqn_hard.pth')
        print(f"Loaded {args.difficulty} model")
        log_info(f"Loaded {args.difficulty} model")
    except FileNotFoundError:
        print(f"No {args.difficulty} model found. Training new model...")
        log_info(f"No {args.difficulty} model found. Training new model...")
        agent = train(computer_player, render_gui)

    test_env = ChineseChessEnv(computer_player=computer_player, agent=agent)
    test(agent, test_env)