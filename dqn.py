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
    def __init__(self, state_dim, action_dim, batch_size=32, gamma=0.99, learning_rate=0.001, memory_size=10000):
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.memory = []
        self.losses = []

        # 神经网络结构
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def select_action(self, state, epsilon, env):
        if np.random.random() < epsilon:
            legal_moves = env.get_legal_moves()
            if not legal_moves:
                log_warning("No legal moves available, returning None")
                return None
            return legal_moves[np.random.randint(len(legal_moves))]

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.q_network.device)
            q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
            legal_moves = env.get_legal_moves()
            if not legal_moves:
                log_warning("No legal moves available during greedy selection")
                return None

            # 确保 Q 值与合法动作对应
            valid_q_values = [q_values[i] for i, move in enumerate(env.action_space) if move in legal_moves]
            if not valid_q_values:
                log_warning("No valid Q values for legal moves")
                return legal_moves[np.random.randint(len(legal_moves))]
            best_action_idx = np.argmax(valid_q_values)
            return legal_moves[best_action_idx]

    def store_transition(self, state, action, reward, next_state, done, env):
        transition = (state, action, reward, next_state, done)
        self.memory.append(transition)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        log_info(f"Stored transition: Action={action}, Reward={reward}, Memory size={len(self.memory)}")

    def update(self, env):
        if len(self.memory) < self.batch_size:
            log_warning(f"Buffer is empty at step, memory size={len(self.memory)}, batch_size={self.batch_size}")
            return

        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in batch])

        states = torch.FloatTensor(np.array(states)).to(self.q_network.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.q_network.device)
        rewards = torch.FloatTensor(rewards).to(self.q_network.device)
        dones = torch.FloatTensor(dones).to(self.q_network.device)

        # 转换为动作索引
        legal_moves = env.get_legal_moves()
        action_indices = [legal_moves.index(action) if action in legal_moves else -1 for action in actions]
        actions = torch.LongTensor([i for i in action_indices if i != -1]).to(self.q_network.device)
        if len(actions) == 0:
            log_warning("No valid action indices found for batch")
            return

        # 计算 Q 值
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, targets)
        self.losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期更新目标网络
        if len(self.memory) % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            log_info("Target network updated")

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)
        log_info(f"Model saved to {path}")

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.q_network.eval()
        self.target_network.eval()
        log_info(f"Model loaded from {path}")