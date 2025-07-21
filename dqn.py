import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging

# 配置日志
logging.basicConfig(filename='training.log', level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def log_info(message):
    logger.info(message)


def log_warning(message):
    logger.warning(message)


def log_error(message):
    logger.error(message)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, batch_size=32, gamma=0.99, learning_rate=0.001, memory_size=10000,
                 target_update_freq=100):
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.target_update_freq = target_update_freq
        self.memory = []
        self.losses = []
        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = self._build_network(action_dim)
        self.target_network = self._build_network(action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def _build_network(self, action_dim):
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(self.device)

    def select_action(self, state, legal_moves, epsilon=0.1):
        """
        Select an action using epsilon-greedy policy
        """
        if not legal_moves:
            log_warning("No legal moves available, returning None")
            return None

        if np.random.random() < epsilon:
            return legal_moves[np.random.randint(len(legal_moves))]

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()

            # Handle case where q_values is empty
            if len(q_values) == 0:
                log_warning("Empty Q-values array, selecting random move")
                return legal_moves[np.random.randint(len(legal_moves))]

            # Create a mapping from action indices to legal moves
            valid_q_values = np.zeros(len(legal_moves))
            for i, move in enumerate(legal_moves):
                # Use a simple hash to map moves to Q-value indices
                move_hash = hash(move) % len(q_values)
                valid_q_values[i] = q_values[move_hash]

            best_action_idx = np.argmax(valid_q_values)
            return legal_moves[best_action_idx]

    def store_transition(self, state, action, reward, next_state, done):
        """
        Simplified to not require env parameter
        """
        transition = (state, action, reward, next_state, done)
        self.memory.append(transition)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        log_info(f"Stored transition: Action={action}, Reward={reward}, Memory size={len(self.memory)}")

    def update(self):
        """
        Simplified to not require env parameter
        """
        self.steps_done += 1
        if len(self.memory) < self.batch_size:
            log_warning(f"Not enough samples for update, memory size={len(self.memory)}")
            return

        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in batch])

        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Convert actions to indices (assuming actions are already indices)
        actions = torch.LongTensor(actions).to(self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(current_q, target_q)
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            log_info(f"Target network updated at step {self.steps_done}")

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)
        log_info(f"Model saved to {path}")

    def load(self, path):
        try:
            self.q_network.load_state_dict(torch.load(path))
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.q_network.eval()
            self.target_network.eval()
            log_info(f"Model loaded from {path}")
        except FileNotFoundError:
            log_warning(f"Model file not found at {path}, skipping load")