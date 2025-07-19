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

    def select_action(self, state, epsilon, env):
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            log_warning("No legal moves available, returning None")
            return None

        if np.random.random() < epsilon:
            return legal_moves[np.random.randint(len(legal_moves))]

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
            if len(legal_moves) > len(q_values):
                q_values = np.pad(q_values, (0, len(legal_moves) - len(q_values)), mode='constant',
                                  constant_values=np.min(q_values))
            valid_q_values = [q_values[i] for i in range(len(legal_moves))]
            best_action_idx = np.argmax(valid_q_values)
            return legal_moves[best_action_idx]

    def store_transition(self, state, action, reward, next_state, done, env):
        transition = (state, action, reward, next_state, done)
        self.memory.append(transition)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        log_info(f"Stored transition: Action={action}, Reward={reward}, Memory size={len(self.memory)}")

    def update(self, env):
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

        legal_moves = env.get_legal_moves()
        if not legal_moves:
            log_warning("No legal moves during update")
            return

        action_indices = []
        for action in actions:
            try:
                idx = legal_moves.index(action)
            except ValueError:
                idx = np.random.randint(len(legal_moves))
                log_warning(f"Invalid action {action}, assigned random index {idx}")
            action_indices.append(idx)
        actions = torch.LongTensor(action_indices).to(self.device)

        q_values = self.q_network(states)
        if q_values.size(1) < actions.max() + 1:
            pad_size = actions.max() + 1 - q_values.size(1)
            q_values = torch.cat([q_values, torch.full((q_values.size(0), pad_size), float('-inf')).to(self.device)], dim=1)

        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            if next_q_values.size(1) == 0:
                log_warning("next_q_values is empty, skipping update")
                return
            max_next_q = next_q_values.max(1)[0]
            targets = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.criterion(q_values, targets)
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
