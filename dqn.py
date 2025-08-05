import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, List, Dict, Any  # 添加标准类型导入


class ChessDQN(nn.Module):
    def __init__(self, input_dim=1530):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10 * 9 * 10 * 9)  # 所有可能移动
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.fc(x)


class DQNAgent:
    def __init__(self, env, difficulty='medium', batch_size=64, gamma=0.99,
                 lr=0.00025, memory_size=100000, target_update=5000,
                 use_amp=True, fast_mode=False):
        self.use_amp = use_amp and torch.cuda.is_available()

        # 移除原有的scaler创建，改为条件创建
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None  # 不使用混合精度
        self.batch_size = batch_size
        self.memory_size = max(memory_size, batch_size * 10)  # 确保足够大的内存
        self.warmup_steps = max(batch_size * 5, 1000)  # 更合理的预热步数
        """优化后的初始化方法"""
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 训练参数优化
        self.gamma = gamma
        self.lr = lr
        self.target_update = target_update
        self.fast_mode = fast_mode
        self.use_amp = use_amp

        # 探索策略优化
        self.difficulty_settings = {
            'easy': {'epsilon': 0.9, 'decay': 0.995, 'min': 0.1},
            'medium': {'epsilon': 0.7, 'decay': 0.997, 'min': 0.05},
            'hard': {'epsilon': 0.5, 'decay': 0.999, 'min': 0.01},
            'expert': {'epsilon': 0.3, 'decay': 0.9995, 'min': 0.005}
        }
        cfg = self.difficulty_settings[difficulty]
        self.epsilon = cfg['epsilon']
        self.epsilon_decay = cfg['decay']
        self.epsilon_min = cfg['min']

        # 网络架构优化
        self.q_net = nn.Sequential(
            nn.Linear(env.state_dim, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),  # 添加层标准化
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 10 * 9 * 10 * 9)  # 动作空间
        ).to(self.device)

        self.target_net = nn.Sequential(
            nn.Linear(env.state_dim, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 10 * 9 * 10 * 9)
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # 优化器配置优化
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.SmoothL1Loss()

        # 经验回放优化
        self.memory = deque(maxlen=memory_size)
        self.steps = 0
        self.update_count = 0

    def select_action(self, state, legal_moves, epsilon=None):
        """改进的ε-greedy策略"""
        if not legal_moves:
            return None

        epsilon = epsilon if epsilon is not None else self.epsilon

        # 随机探索
        if random.random() < epsilon:
            return random.choice(legal_moves)

        # 确保状态有效性
        if not isinstance(state, np.ndarray):
            state = self.env.get_state()
        state = state.reshape(-1)[:self.env.state_dim]

        # 使用网络预测
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor).cpu().numpy().flatten()

        # 创建合法动作掩码
        legal_mask = np.zeros_like(q_values, dtype=bool)
        for move in legal_moves:
            try:
                idx = self._move_to_index(move)
                legal_mask[idx] = True
            except Exception as e:
                print(f"非法移动转换: {move}, 错误: {str(e)}")
                continue

        # 过滤非法动作
        valid_q = np.where(legal_mask, q_values, -np.inf)
        best_idx = np.argmax(valid_q)

        # 转换回移动
        try:
            return self._index_to_move(best_idx)
        except:
            print(f"select action failure")
            return random.choice(legal_moves)  # 失败时随机选择

    def store_transition(self, state, action, reward, next_state, done):
        """严格验证的经验存储"""
        assert isinstance(state, np.ndarray), f"State must be numpy array, got {type(state)}"
        assert isinstance(action, tuple) and len(action) == 4, f"Invalid action format: {action}"
        assert isinstance(done, bool), f"Done must be boolean, got {type(done)}"

        try:
            action_idx = self._move_to_index(action)
        except Exception as e:
            print(f"无法存储转换: 动作{action}转换失败: {str(e)}")
            return

        # 标准化处理
        state = state.astype(np.float32).reshape(-1)[:self.env.state_dim]
        next_state = next_state.astype(np.float32).reshape(-1)[:self.env.state_dim] if next_state is not None else None
        reward = float(np.clip(reward, -1, 1))

        self.memory.append((
            state,
            action_idx,
            reward,
            next_state,
            done
        ))

    def update(self):
        """优化后的网络更新方法，增加梯度监控和稳定性检查"""
        if len(self.memory) < self.batch_size or self.steps < self.warmup_steps:
            return None

        try:
            # 1. 采样批次并验证数据完整性
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # 数据验证和预处理优化
            states = np.asarray(states, dtype=np.float32)  # 显式指定类型
            if states.shape[1] != self.env.state_dim:
                raise ValueError(f"状态维度不匹配: 预期{self.env.state_dim}, 实际{states.shape[1]}")

            # 2. 使用torch.as_tensor避免内存复制
            states = torch.as_tensor(states, device=self.device)
            actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
            dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device).unsqueeze(1)

            # 3. 优化next_states处理 (修复警告的关键修改)
            non_final_mask = torch.tensor(
                [s is not None for s in next_states],
                dtype=torch.bool, device=self.device
            )

            # 预分配内存的优化方案
            valid_next_states = [s for s in next_states if s is not None]
            if valid_next_states:
                # 使用stack代替列表推导
                non_final_next_states = torch.stack(
                    [torch.as_tensor(s, device=self.device) for s in valid_next_states]
                )
            else:
                non_final_next_states = torch.empty(
                    (0, self.env.state_dim),
                    dtype=torch.float32,
                    device=self.device
                )

            # 4. 自动混合精度上下文管理优化
            autocast_context = torch.amp.autocast(
                device_type=self.device.type,  # 自动适配设备类型
                enabled=self.use_amp and self.device.type == 'cuda'  # 双重检查
            )

            with autocast_context:
                current_q = self.q_net(states).gather(1, actions)

                # 5. 计算目标Q值 (使用原地操作优化)
                next_q = torch.zeros_like(current_q)
                if non_final_next_states.shape[0] > 0:
                    with torch.no_grad():  # 目标网络不需要梯度
                        next_q[non_final_mask] = self.target_net(non_final_next_states).max(1, keepdim=True)[0]

                target_q = rewards + self.gamma * next_q * (~dones)
                target_q.clamp_(-1.0, 1.0)  # 原地操作

                # 6. 使用更稳定的损失计算
                loss = self.criterion(current_q.float(), target_q.float())  # 确保精度一致

            # 7. 梯度管理优化
            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                # 梯度裁剪前取消缩放
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
                self.optimizer.step()

            # 8. 参数更新
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.steps += 1

            # 9. 目标网络更新 (添加状态验证)
            if self.steps % self.target_update == 0:
                with torch.no_grad():
                    for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
                        target_param.data.copy_(param.data)

            return {
                'loss': loss.item(),
                'grad_norm': grad_norm.item(),
                'max_q': current_q.max().item(),
                'target_q': target_q.mean().item()
            }

        except Exception as e:
            print(f"更新失败: {str(e)}")
            # 重置网络状态防止污染
            self._safe_reset_networks()
            return None

    def _safe_reset_networks(self):
        """安全重置网络状态"""
        with torch.no_grad():
            for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(param.data)

    def save(self, path):
        """改进的模型保存"""
        torch.save({
            'q_net_state': self.q_net.state_dict(),
            'target_net_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)

    def load(self, path):
        """改进的模型加载"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state'])
        self.target_net.load_state_dict(checkpoint['target_net_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.steps = checkpoint.get('steps', 0)

    def _move_to_index(self, move):
        """
        将象棋移动(i,j,ni,nj)转换为线性索引
        参数:
            move: 元组 (i, j, ni, nj)
                i,j - 起始行列 (0-9, 0-8)
                ni,nj - 目标行列
        返回:
            线性索引 (0-8099)
        """
        i, j, ni, nj = move
        # 棋盘是10行(x)9列(y)
        # 索引计算: (i*9 + j)*90 + (ni*9 + nj)
        return (i * 9 + j) * (10 * 9) + (ni * 9 + nj)

    def _index_to_move(self, index):
        """
        将线性索引转换回象棋移动
        参数:
            index: 线性索引 (0-8099)
        返回:
            (i, j, ni, nj)
        """
        total_positions = 10 * 9
        from_pos = index // total_positions
        to_pos = index % total_positions
        i, j = divmod(from_pos, 9)
        ni, nj = divmod(to_pos, 9)
        return (i, j, ni, nj)