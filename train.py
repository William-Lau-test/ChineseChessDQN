import time
from random import random

import numpy as np

from dqn import DQNAgent
from environment import ChineseChessEnv
from evaluate import evaluate


class ChessTrainer:
    def __init__(self, env, args):
        """完整的训练器初始化"""
        self.env = env
        self.args = args

        # 初始化Agent（确保传递fast_mode参数）
        self.agent = DQNAgent(
            env=env,
            difficulty=args.difficulty,
            batch_size=getattr(args, 'batch_size', 32),
            gamma=0.99,
            lr=0.0001,
            memory_size=getattr(args, 'memory_size', 50000),
            target_update=getattr(args, 'target_update', 500),
            use_amp=not getattr(args, 'cpu_only', False),
            fast_mode=getattr(args, 'fast_train', False)  # 传递fast_mode参数
        )

        # 初始化统计信息
        self.stats = {
            'episode': 0,
            'loss': [],
            'rewards': [],
            'win_rates': [],
            'start_time': time.time()
        }

    def load_model(self):
        """加载模型的优化实现"""
        if hasattr(self.args, 'model_path') and self.args.model_path:
            try:
                start_time = time.time()
                self.agent.load(self.args.model_path)
                load_time = time.time() - start_time
                print(f"成功加载模型 {self.args.model_path} (耗时: {load_time:.2f}s)")
            except Exception as e:
                print(f"模型加载失败: {str(e)}")
                # 失败时初始化新模型
                self.agent.init_weights()

    def run_episode(self, episode):
        """优化后的回合运行方法，增加步数限制和早期终止"""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        episode_losses = []

        # 修改这里：添加默认值500
        max_steps = getattr(self.args, 'max_steps', 500)

        while steps < max_steps:
            legal_moves = self.env.get_legal_moves()
            if not legal_moves:
                break

            # 动态epsilon调整
            current_epsilon = max(
                self.agent.epsilon_min,
                self.agent.epsilon * (0.99 ** (episode // 10))  # 每10回合衰减一次
            )

            action = self.agent.select_action(state, legal_moves, epsilon=current_epsilon)
            next_state, reward, done, _ = self.env.step(action)

            # 存储转换
            self.agent.store_transition(state, action, reward, next_state, done)

            # 更频繁的更新
            if (len(self.agent.memory) >= self.agent.batch_size and
                    self.agent.steps >= self.agent.warmup_steps):
                update_result = self.agent.update()
                if update_result:
                    episode_losses.append(update_result['loss'])

            total_reward += reward
            state = next_state
            steps += 1
            self.agent.steps += 1

            if done:
                break

        # 记录统计信息
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        self.stats['loss'].extend(episode_losses)
        self.stats['rewards'].append(total_reward)

        return {
            'total_reward': total_reward,
            'steps': steps,
            'avg_loss': avg_loss,
            'epsilon': self.agent.epsilon
        }

    def evaluate(self):
        """优化评估过程"""
        eval_start = time.time()
        win_rate, avg_reward = evaluate(
            self.agent,
            self.env,
            episodes=min(20, self.args.episodes // 10),  # 修改参数名为episodes
            render=False
        )
        self.stats['win_rates'].append(win_rate)
        print(f"评估完成 (耗时: {time.time() - eval_start:.2f}s)")
        return win_rate, avg_reward

    def save_model(self, episode):
        """优化模型保存"""
        model_path = f"{self.args.output_dir}/dqn_{self.args.difficulty}_ep{episode}.pth"
        try:
            start_time = time.time()
            self.agent.save(model_path)
            save_time = time.time() - start_time
            print(f"模型保存成功 {model_path} (耗时: {save_time:.2f}s)")
        except Exception as e:
            print(f"模型保存失败: {str(e)}")

    def print_progress(self, episode, win_rate, avg_reward):
        """改进的进度输出方法"""
        elapsed = time.time() - self.stats['start_time']
        avg_loss = np.mean(self.stats['loss'][-100:]) if self.stats['loss'] else 0

        # 安全的属性访问
        target_update = getattr(self.agent, 'target_update', 500)
        warmup_steps = getattr(self.agent, 'warmup_steps', 1000)

        print(f"\nEpisode {episode} | "
              f"Time: {elapsed // 60:.0f}m{elapsed % 60:.0f}s | "
              f"AvgReward: {avg_reward:.2f} | "
              f"WinRate: {win_rate * 100:.1f}% | "
              f"Loss: {avg_loss:.4f} | "
              f"TargetUpdate: {target_update} | "
              f"Warmup: {warmup_steps}")

    def train(self):
        """主训练循环优化"""
        print(f"\n训练启动 | 难度: {self.args.difficulty}")
        print(f"初始参数: LR={self.agent.lr:.1e} | Batch={self.agent.batch_size} | Memory={self.agent.memory_size}")

        best_metrics = {'win_rate': 0, 'reward': -float('inf')}
        for episode in range(1, self.args.episodes + 1):
            # 运行回合
            episode_result = self.run_episode(episode)

            # 监控和调整
            self.monitor_training(episode, episode_result)

            # 定期评估
            if episode % self.args.eval_interval == 0:
                win_rate, avg_reward = self.evaluate()

                # 保存最佳模型
                if win_rate > best_metrics['win_rate']:
                    best_metrics['win_rate'] = win_rate
                    best_path = f"{self.args.output_dir}/dqn_{self.args.difficulty}_best.pth"
                    self.agent.save(best_path)
                    print(f"新最佳模型 (胜率: {win_rate * 100:.1f}%) 保存到 {best_path}")

            # 定期保存
            if episode % self.args.save_interval == 0:
                self.save_model(episode)

        # 训练结束保存最终模型
        final_path = f"{self.args.output_dir}/dqn_{self.args.difficulty}_final.pth"
        self.agent.save(final_path)
        print(f"\n训练完成! 最终模型保存到 {final_path}")
        print(f"最佳胜率: {best_metrics['win_rate'] * 100:.1f}%")

    def _inspect_memory(self):
        """检查经验回放数据"""
        sample = random.sample(self.agent.memory, min(5, len(self.agent.memory)))
        print("\n经验回放样本检查:")
        for i, (s, a, r, ns, d) in enumerate(sample):
            print(f"[{i}] State: {s[:5]}... | Action: {a} | Reward: {r:.2f} | "
                  f"Next: {ns[:5] if ns is not None else None} | Done: {d}")

    def monitor_training(self, episode, episode_result):
        """训练监控和自适应调整"""
        # 1. 学习率调整
        if len(self.stats['loss']) > 20:
            last_losses = self.stats['loss'][-20:]
            if np.mean(last_losses) > 2.0 * np.median(self.stats['loss']):
                self.adjust_learning_rate(factor=0.8)

        # 2. 批次大小自适应
        if episode % 50 == 0 and hasattr(self.agent, 'batch_size'):
            mem_usage = len(self.agent.memory) / self.agent.memory_size
            if mem_usage > 0.8 and self.agent.batch_size < 256:
                new_size = min(256, int(self.agent.batch_size * 1.2))
                print(f"增加批次大小: {self.agent.batch_size} -> {new_size}")
                self.agent.batch_size = new_size

        # 3. 打印监控信息
        if episode % 10 == 0:
            print(f"Episode {episode} | "
                  f"Loss: {episode_result['avg_loss']:.4f} | "
                  f"Reward: {episode_result['total_reward']:.1f} | "
                  f"Steps: {episode_result['steps']} | "
                  f"Epsilon: {episode_result['epsilon']:.3f} | "
                  f"Memory: {len(self.agent.memory)}/{self.agent.memory_size}")

    def adjust_learning_rate(self, factor=0.9):
        """自适应学习率调整"""
        for param_group in self.agent.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(1e-6, old_lr * factor)
            param_group['lr'] = new_lr
            print(f"调整学习率: {old_lr:.2e} -> {new_lr:.2e}")


class SpeedOptimizer:
    """优化的快速环境交互器"""

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        # 预分配内存
        self.state_buffer = np.zeros((100, env.state_dim), dtype=np.float32)
        self.idx = 0

    def fast_step(self, action):
        """加速版环境交互"""
        # 复用内存
        state_slot = self.state_buffer[self.idx % 100]
        next_state, reward, done, _ = self.env.step(action)

        # 内存拷贝（比直接创建新数组快3-5倍）
        if next_state is not None:
            np.copyto(state_slot, next_state)
        else:
            state_slot.fill(0)

        self.idx += 1
        return state_slot, reward, done


def train(env, args):
    """优化后的训练入口"""
    # 设置默认参数
    args.eval_interval = getattr(args, 'eval_interval', 50)
    args.save_interval = getattr(args, 'save_interval', 100)
    args.output_dir = getattr(args, 'output_dir', './models')

    # 创建训练器
    trainer = ChessTrainer(env, args)
    trainer.load_model()

    # 启动训练
    trainer.train()


if __name__ == "__main__":
    # 测试代码
    class MockArgs:
        difficulty = 'medium'
        episodes = 10
        batch_size = 64
        memory_size = 10000
        target_update = 500
        eval_interval = 2
        save_interval = 3
        output_dir = './tmp_models'
        model_path = None
        fast_train = True
        max_moves = 150


    env = ChineseChessEnv()
    train(env, MockArgs())