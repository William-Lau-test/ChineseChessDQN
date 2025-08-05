import argparse
import time
import pygame
from environment import ChineseChessEnv
from dqn import DQNAgent
from train import train
from evaluate import evaluate

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated as an API")

def parse_args():
    """优化后的参数解析器，包含训练加速参数"""
    parser = argparse.ArgumentParser(description="中国象棋DQN智能体")

    # 模式选择
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument('--train', action='store_true', help="训练模式")
    mode_group.add_argument('--eval', action='store_true', help="评估模式")
    mode_group.add_argument('--ai-vs-ai', action='store_true', help="AI对战模式")

    # 训练加速参数
    parser.add_argument('--fast-train', action='store_true',
                        help="启用快速训练模式(禁用渲染，减小批次)")
    parser.add_argument('--cpu-only', action='store_true',
                        help="强制使用CPU(避免CUDA内存问题)")

    # 通用参数
    parser.add_argument('--difficulty', default='medium',
                        choices=['easy', 'medium', 'hard', 'expert'],
                        help="AI难度级别")
    parser.add_argument('--render', action='store_true',
                        help="启用可视化(训练时不建议开启)")

    # 训练专用参数 (优化默认值)
    parser.add_argument('--episodes', type=int, default=1000,
                        help="训练回合数")
    parser.add_argument('--batch-size', type=int, default=32 if not parser.get_default('fast_train') else 16,
                        help="训练批次大小")
    parser.add_argument('--memory-size', type=int, default=50000,
                        help="经验回放缓存大小")
    parser.add_argument('--target-update', type=int, default=500,
                        help="目标网络更新频率")
    parser.add_argument('--render-interval', type=int, default=100,
                        help="训练时渲染间隔")
    parser.add_argument('--eval-interval', type=int, default=100,
                        help="评估间隔回合数")
    parser.add_argument('--save-interval', type=int, default=100,
                        help="模型保存间隔")
    parser.add_argument('--output-dir', type=str, default='./models',
                        help="模型输出目录")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="学习率(原1e-3太大)")
    parser.add_argument('--max-moves', type=int, default=200,
                      help="单回合最大步数")
    parser.add_argument('--warmup-steps', type=int, default=100,
                      help="预热步数不更新网络")
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epsilon-decay', type=float, default=0.998)

    # 对战模式参数
    parser.add_argument('--human-side', choices=['red', 'black'], default='red',
                        help="人类玩家执方")
    parser.add_argument('--move-delay', type=float, default=0.3,
                        help="AI走棋延迟(秒)")

    parser.add_argument('--model-path', type=str, default=None,
                        help="预训练模型路径")

    parser.set_defaults(
        batch_size=64 if not parser.get_default('fast_train') else 128,
        memory_size=100000,
        target_update=1000,
        episodes=5000,
        lr=0.00025  # 优化后的学习率
    )

    args = parser.parse_args()
    if not any([args.train, args.eval, args.ai_vs_ai]):
        args.render = True  # 人机对战强制开启渲染
        args.difficulty = 'medium'
        args.human_side = 'red'
        print("\n未指定模式，自动启动人机对战：")
        print(f"  人类执红方 | AI难度: {args.difficulty} | 渲染: {'开启' if args.render else '关闭'}")

    return args


def human_vs_ai(env, human_side, difficulty, move_delay=0.5):
    """优化后的人机对战函数"""
    agent = DQNAgent(env, difficulty=difficulty)
    state = env.reset()

    while True:
        env.render_pygame()

        if env.current_player == (1 if human_side == 'red' else -1):
            action = env.get_move_from_clicks()
            if action is None:
                break
        else:
            legal_moves = env.get_legal_moves()
            action = agent.select_action(state, legal_moves, epsilon=0.01)
            time.sleep(move_delay)

        state, _, done, _ = env.step(action)
        if done:
            winner = "红方" if env.current_player == -1 else "黑方"
            print(f"{winner}获胜！")
            break


def main():
    args = parse_args()

    # 快速训练模式自动设置
    if args.fast_train:
        args.render = False
        args.batch_size = 16
        args.memory_size = 25000
        args.target_update = 250
        args.eval_interval = 25

    env = ChineseChessEnv()

    try:
        if args.train:
            # 训练模式
            train(env, args)
        elif args.ai_vs_ai:
            # AI对战模式
            red_ai = DQNAgent(env, difficulty=args.difficulty)
            black_ai = DQNAgent(env, difficulty=args.difficulty)

            state = env.reset()
            while True:
                if args.render:
                    env.render_pygame()
                    time.sleep(args.move_delay)

                current_ai = red_ai if env.current_player == 1 else black_ai
                legal_moves = env.get_legal_moves()
                action = current_ai.select_action(state, legal_moves, epsilon=0.01)
                state, _, done, _ = env.step(action)

                if done:
                    winner = "红方" if env.current_player == -1 else "黑方"
                    print(f"{winner}获胜！")
                    break
        elif args.eval:
            # 评估模式
            agent = DQNAgent(env, difficulty=args.difficulty)
            win_rate, avg_reward = evaluate(agent, env)
            print(f"评估结果 - 胜率: {win_rate:.2%} | 平均奖励: {avg_reward:.2f}")
        else:
            # 默认人机对战
            human_vs_ai(env, args.human_side, args.difficulty, args.move_delay)

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        env.close()


if __name__ == "__main__":
    main()