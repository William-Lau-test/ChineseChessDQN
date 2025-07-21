import argparse
import numpy as np
import matplotlib.pyplot as plt
import pygame
from environment import ChineseChessEnv
from dqn import DQN
import logging
from collections import deque
import time
import csv
import os

# Configure logging
logging.basicConfig(filename='training.log', level=logging.INFO,
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


# Hyperparameters
EPISODES = 8000
MAX_STEPS = 200
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 5000
RENDER_GUI = True
BATCH_SIZE = 128
WARMUP_STEPS = 1000
TARGET_UPDATE_FREQ = 100
MEMORY_SIZE = 10000

# Difficulty settings
DIFFICULTY_SETTINGS = {
    'easy': {'epsilon': 0.3, 'gamma': 0.9, 'learning_rate': 0.0005},
    'medium': {'epsilon': 0.2, 'gamma': 0.95, 'learning_rate': 0.001},
    'hard': {'epsilon': 0.1, 'gamma': 0.99, 'learning_rate': 0.001},
    'expert': {'epsilon': 0.05, 'gamma': 0.995, 'learning_rate': 0.0005}
}


def parse_args():
    parser = argparse.ArgumentParser(description="DQN for Chinese Chess")
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard', 'expert'],
                        default='hard', help='AI difficulty level')
    parser.add_argument('--render-gui', action='store_true', default=RENDER_GUI,
                        help='Render GUI during training')
    parser.add_argument('--ai-vs-ai', action='store_true', help='Enable AI vs AI mode')
    parser.add_argument('--train', action='store_true', help='Train the model')
    return parser.parse_args()


def get_player_side():
    while True:
        side = input("Choose your side (red/black): ").strip().lower()
        if side in ['red', 'black']:
            return 1 if side == 'red' else -1
        print("Invalid choice. Please enter 'red' or 'black'.")


def plot_training_metrics(episode_rewards, episode_losses, invalid_actions, win_rates):
    """Plot and save training metrics with smoothed curves"""
    plt.figure(figsize=(15, 10))

    # Smoothing function
    def smooth(data, weight=0.9):
        smoothed = []
        last = data[0] if len(data) > 0 else 0
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    # Plot rewards
    plt.subplot(2, 2, 1)
    if len(episode_rewards) > 0:
        plt.plot(episode_rewards, alpha=0.3, label='Raw')
        plt.plot(smooth(episode_rewards), label='Smoothed')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    # Plot losses
    plt.subplot(2, 2, 2)
    if len(episode_losses) > 0:
        plt.plot(episode_losses, alpha=0.3, label='Raw')
        plt.plot(smooth(episode_losses), label='Smoothed')
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()

    # Plot invalid actions
    plt.subplot(2, 2, 3)
    if len(invalid_actions) > 0:
        plt.plot(invalid_actions, alpha=0.3, label='Raw')
        plt.plot(smooth(invalid_actions), label='Smoothed')
    plt.title('Invalid Actions')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.legend()

    # Plot win rates
    plt.subplot(2, 2, 4)
    if len(win_rates) > 0:
        plt.plot(win_rates, alpha=0.3, label='Raw')
        plt.plot(smooth(win_rates), label='Smoothed')
    plt.title('Win Rate (Last 100 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def init_csv_log():
    """Initialize CSV log file with headers"""
    if not os.path.exists('training_log.csv'):
        with open('training_log.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'loss', 'invalid_actions',
                             'win_rate', 'epsilon', 'timestamp'])


def log_training_data(episode, reward, loss, invalid_actions, win_rate, epsilon):
    """Log training data to CSV"""
    with open('training_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            episode,
            reward,
            loss,
            invalid_actions,
            win_rate,
            epsilon,
            time.strftime('%Y-%m-%d %H:%M:%S')
        ])


def ai_vs_ai(agent1, agent2=None, render_gui=True):
    env = ChineseChessEnv(computer_player=1, agent=agent1)
    state = env.reset()
    done = False

    # If only one agent provided, use it for both sides
    if agent2 is None:
        agent2 = agent1

    while not done:
        if render_gui:
            env.render_pygame()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

        legal_moves = env.get_legal_moves()
        if not legal_moves:
            print("No legal moves available!")
            break

        # Select agent based on current player
        current_agent = agent1 if env.current_player == 1 else agent2
        action = current_agent.select_action(state, legal_moves)

        if action is None:
            print("No action selected!")
            break

        state, reward, done, status = env.step(action)

        print(f"Status: {status}, Reward: {reward}")

        if render_gui:
            pygame.time.delay(500)  # Slow down for visualization

        if done:
            if status == "checkmate":
                winner = "Red" if reward > 0 else "Black"
                print(f"Game over! {winner} wins by checkmate!")
            elif status == "draw_80_moves":
                print("Game ended in a draw by 80-move rule")
            break

    env.close()


def train(difficulty='hard', render_gui=RENDER_GUI):
    # Initialize logging
    init_csv_log()

    env = ChineseChessEnv(computer_player=1)
    state_dim = env.state_dim
    action_dim = len(env.get_legal_moves())

    # Get difficulty settings
    settings = DIFFICULTY_SETTINGS[difficulty]

    agent = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        batch_size=BATCH_SIZE,
        gamma=settings['gamma'],
        learning_rate=settings['learning_rate'],
        memory_size=MEMORY_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ
    )

    rewards = []
    avg_losses = []
    invalid_actions = []
    win_rates = []
    recent_wins = deque(maxlen=100)
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        state = env.reset()
        ep_reward, ep_losses, ep_invalid = 0, [], 0
        done = False

        if render_gui:
            env.render_pygame()

        for t in range(MAX_STEPS):
            legal_moves = env.get_legal_moves()
            if not legal_moves:
                break

            action = agent.select_action(state, legal_moves, epsilon)
            if action is None:
                break

            next_state, reward, done, status = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Update agent
            if len(agent.memory) > WARMUP_STEPS:
                agent.update()

            state = next_state
            ep_reward += reward

            if reward <= -1.0:  # Penalty for invalid moves
                ep_invalid += 1

            if agent.losses:
                ep_losses.append(agent.losses[-1])

            if render_gui:
                env.render_pygame()
                pygame.time.delay(50)

            if done:
                if status == "checkmate":
                    recent_wins.append(1 if reward > 0 else 0)
                break

        # Calculate metrics
        avg_loss = np.mean(ep_losses) if ep_losses else 0
        win_rate = np.mean(recent_wins) if len(recent_wins) > 0 else 0

        # Store metrics
        rewards.append(ep_reward)
        avg_losses.append(avg_loss)
        invalid_actions.append(ep_invalid)
        if len(recent_wins) > 0:
            win_rates.append(win_rate)

        # Log to CSV
        log_training_data(episode, ep_reward, avg_loss, ep_invalid, win_rate, epsilon)

        # Decay epsilon
        epsilon = max(settings['epsilon'], EPSILON_START * np.exp(-episode / EPSILON_DECAY))

        # Print progress
        if episode % 100 == 0:
            log_info(f"Episode {episode}, Reward: {ep_reward:.2f}, Loss: {avg_loss:.4f}, "
                     f"Invalid: {ep_invalid}, Win Rate: {win_rate:.2f}, Epsilon: {epsilon:.2f}")
            print(f"Episode {episode}, Reward: {ep_reward:.2f}, Loss: {avg_loss:.4f}, "
                  f"Win Rate: {win_rate:.2f}")

        # Save model and plot periodically
        if episode % 500 == 0 or episode == EPISODES - 1:
            agent.save(f'dqn_{difficulty}_ep{episode}.pth')
            plot_training_metrics(rewards, avg_losses, invalid_actions, win_rates)

    # Final save
    agent.save(f'dqn_{difficulty}.pth')
    plot_training_metrics(rewards, avg_losses, invalid_actions, win_rates)

    env.close()
    return agent


def test(agent, difficulty='hard', human_player=1, render_gui=True):
    computer_player = -human_player
    env = ChineseChessEnv(computer_player=computer_player, agent=agent)
    test_rewards = []
    wins = 0

    for episode in range(5):  # Test with 5 games
        state = env.reset()
        episode_reward = 0
        done = False

        if render_gui:
            env.render_pygame()

        while not done:
            if env.current_player == computer_player:
                legal_moves = env.get_legal_moves()
                action = agent.select_action(state, legal_moves,
                                             epsilon=DIFFICULTY_SETTINGS[difficulty]['epsilon'])
            else:
                action = env.get_move_from_clicks()
                if action is None:
                    continue

            next_state, reward, done, status = env.step(action)
            state = next_state
            episode_reward += reward

            if render_gui:
                env.render_pygame()
                pygame.time.delay(300)

            if done:
                if status == "checkmate":
                    if (reward > 0 and human_player == 1) or (reward < 0 and human_player == -1):
                        wins += 1
                print(f"Game {episode + 1} result: {status}")
                pygame.time.delay(2000)
                break

        test_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}, Reward: {episode_reward:.2f}")

    print(f"Test Results - Win Rate: {wins / 5:.2f}, Mean Reward: {np.mean(test_rewards):.2f}")
    env.close()


if __name__ == "__main__":
    args = parse_args()

    if args.train:
        print(f"Training {args.difficulty} model...")
        agent = train(difficulty=args.difficulty, render_gui=args.render_gui)
    elif args.ai_vs_ai:
        print("Starting AI vs AI mode...")
        env = ChineseChessEnv(computer_player=1)
        agent = DQN(env.state_dim, len(env.get_legal_moves()))
        try:
            agent.load(f'dqn_{args.difficulty}.pth')
            print(f"Loaded {args.difficulty} model successfully")
        except:
            print(f"Model not found. Please train first with --train")
            exit()
        ai_vs_ai(agent, render_gui=args.render_gui)
    else:
        human_player = get_player_side()
        computer_player = -human_player
        env = ChineseChessEnv(computer_player=computer_player)
        agent = DQN(env.state_dim, len(env.get_legal_moves()))
        try:
            agent.load(f'dqn_{args.difficulty}.pth')
            print(f"Loaded {args.difficulty} model successfully")
        except:
            print(f"Model not found. Please train first with --train")
            exit()
        test(agent, difficulty=args.difficulty, human_player=human_player, render_gui=args.render_gui)