import numpy as np
import torch
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

def select_action(agent, state, epsilon, env):
    valid_moves = env.get_legal_moves()  # 每次调用时获取最新动作空间
    if not valid_moves:
        log_error(f"No legal moves available for player {env.current_player}")
        return None

    if np.random.random() < epsilon:
        action = valid_moves[np.random.randint(len(valid_moves))]
        return action

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q_values = agent.q_network(state_tensor).squeeze(0).cpu().numpy()

    if len(q_values) != len(env.action_space):  # 确保 Q 值与动作空间匹配
        log_warning(f"Q-values length ({len(q_values)}) mismatch with action_space size ({len(env.action_space)}), padding to match")
        q_values = np.pad(q_values, (0, len(env.action_space) - len(q_values)), 'constant')

    valid_q_values = [(q_values[i], move) for i, move in enumerate(env.action_space) if move in valid_moves]
    if not valid_q_values:
        action = valid_moves[np.random.randint(len(valid_moves))]
        return action

    best_q, best_action = max(valid_q_values, key=lambda x: x[0])
    return best_action