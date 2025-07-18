import numpy as np
import pygame
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

class ChineseChessEnv:
    def __init__(self, computer_player=1, agent=None):
        self.board = np.zeros((9, 10))
        self.current_player = 1
        self.computer_player = computer_player
        self.agent = agent
        self.state_dim = 90
        self.action_space = self.get_legal_moves()  # 初始动作空间
        self.pygame_initialized = False
        self.piece_values = {
            1: '帅', 2: '仕', 3: '相', 4: '马', 5: '车', 6: '炮', 7: '兵',
            -1: '将', -2: '士', -3: '象', -4: '马', -5: '车', -6: '炮', -7: '卒'
        }
        self.piece_strategic_values = {
            '帅': 10, '将': 10,
            '仕': 2, '士': 2,
            '相': 2, '象': 2,
            '马': 4,
            '车': 9,
            '炮': 4.5,
            '兵': 1, '卒': 1
        }
        log_info(f"Environment initialized with computer_player={computer_player}")

    def reset(self):
        self.board = np.zeros((9, 10))
        # Red pieces
        self.board[0, 0] = 5  # Chariot
        self.board[0, 1] = 4  # Horse
        self.board[0, 2] = 3  # Elephant
        self.board[0, 3] = 2  # Advisor
        self.board[0, 4] = 1  # King
        self.board[0, 5] = 2  # Advisor
        self.board[0, 6] = 3  # Elephant
        self.board[0, 7] = 4  # Horse
        self.board[0, 8] = 5  # Chariot
        self.board[2, 1] = 6  # Cannon
        self.board[2, 7] = 6  # Cannon
        self.board[3, 0] = 7  # Soldier
        self.board[3, 2] = 7  # Soldier
        self.board[3, 4] = 7  # Soldier
        self.board[3, 6] = 7  # Soldier
        self.board[3, 8] = 7  # Soldier
        # Black pieces
        self.board[8, 0] = -5  # Chariot
        self.board[8, 1] = -4  # Horse
        self.board[8, 2] = -3  # Elephant
        self.board[8, 3] = -2  # Advisor
        self.board[8, 4] = -1  # King
        self.board[8, 5] = -2  # Advisor
        self.board[8, 6] = -3  # Elephant
        self.board[8, 7] = -4  # Horse
        self.board[8, 8] = -5  # Chariot
        self.board[6, 1] = -6  # Cannon
        self.board[6, 7] = -6  # Cannon
        self.board[6, 0] = -7  # Pawn
        self.board[6, 2] = -7  # Pawn
        self.board[6, 4] = -7  # Pawn
        self.board[6, 6] = -7  # Pawn
        self.board[6, 8] = -7  # Pawn
        self.current_player = 1
        self.action_space = self.get_legal_moves()
        if not self.action_space:
            log_error("Initial action_space is empty after reset")
        log_info(f"Environment reset with initial board state:\n{self.board}", verbose_only=True)
        return self.get_state()

    def get_state(self):
        return self.board.flatten()

    def get_legal_moves(self):
        legal_moves = []
        for i in range(9):
            for j in range(10):
                if self.board[i, j] * self.current_player > 0:
                    piece = self.board[i, j]
                    moves = []
                    if abs(piece) == 1:  # King
                        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            ni, nj = i + di, j + dj
                            if 3 <= nj <= 5 and ((0 <= ni <= 2) if self.current_player == 1 else (6 <= ni <= 8)):
                                if 0 <= ni < 9 and 0 <= nj < 10 and (
                                        self.board[ni, nj] == 0 or self.board[ni, nj] * self.current_player < 0):
                                    moves.append((i, j, ni, nj))
                    elif abs(piece) == 2:  # Advisor
                        for di, dj in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                            ni, nj = i + di, j + dj
                            if 3 <= nj <= 5 and ((0 <= ni <= 2) if self.current_player == 1 else (6 <= ni <= 8)):
                                if 0 <= ni < 9 and 0 <= nj < 10 and (
                                        self.board[ni, nj] == 0 or self.board[ni, nj] * self.current_player < 0):
                                    moves.append((i, j, ni, nj))
                    elif abs(piece) == 3:  # Elephant
                        for di, dj in [(2, 2), (2, -2), (-2, 2), (-2, -2)]:
                            ni, nj = i + di, j + dj
                            mid_i, mid_j = i + di // 2, j + dj // 2
                            if 0 <= ni < 9 and 0 <= nj < 10 and 0 <= mid_i < 9 and 0 <= mid_j < 10:
                                if self.board[mid_i, mid_j] == 0 and (
                                        self.board[ni, nj] == 0 or self.board[ni, nj] * self.current_player < 0):
                                    moves.append((i, j, ni, nj))
                    elif abs(piece) == 4:  # Horse
                        for di, dj in [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]:
                            ni, nj = i + di, j + dj
                            k, l = i + di // 2, j + dj // 2
                            if 0 <= ni < 9 and 0 <= nj < 10 and 0 <= k < 9 and 0 <= l < 10:
                                if self.board[k, l] == 0 and (
                                        self.board[ni, nj] == 0 or self.board[ni, nj] * self.current_player < 0):
                                    moves.append((i, j, ni, nj))
                    elif abs(piece) == 5:  # Chariot
                        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            ni, nj = i + di, j + dj
                            while 0 <= ni < 9 and 0 <= nj < 10 and self.board[ni, nj] == 0:
                                moves.append((i, j, ni, nj))
                                ni, nj = ni + di, nj + dj
                            if 0 <= ni < 9 and 0 <= nj < 10 and self.board[ni, nj] * self.current_player < 0:
                                moves.append((i, j, ni, nj))
                    elif abs(piece) == 6:  # Cannon
                        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            ni, nj = i + di, j + dj
                            jumped = False
                            while 0 <= ni < 9 and 0 <= nj < 10:
                                if self.board[ni, nj] != 0 and not jumped:
                                    jumped = True
                                elif self.board[ni, nj] != 0 and jumped and self.board[ni, nj] * self.current_player < 0:
                                    moves.append((i, j, ni, nj))
                                    break
                                elif jumped and self.board[ni, nj] == 0:
                                    moves.append((i, j, ni, nj))
                                ni, nj = ni + di, nj + dj
                    elif abs(piece) == 7:  # Pawn
                        ni = i + (1 if self.current_player == 1 else -1)
                        nj = j  # 确保 nj 初始化
                        if 0 <= ni < 9 and 0 <= nj < 10 and (
                                self.board[ni, nj] == 0 or self.board[ni, nj] * self.current_player < 0):
                            moves.append((i, j, ni, nj))
                        if (i >= 5 if self.current_player == 1 else i <= 3):
                            for nj in [j - 1, j + 1]:
                                if 0 <= nj < 10 and (
                                        self.board[i, nj] == 0 or self.board[i, nj] * self.current_player < 0):
                                    moves.append((i, j, i, nj))
                    legal_moves.extend(moves)
        if not legal_moves:
            log_error(f"No legal moves found for player {self.current_player}")
        return legal_moves

    def is_king_threatened(self, player):
        king_pos = None
        for i in range(9):
            for j in range(10):
                if self.board[i, j] == player:
                    king_pos = (i, j)
                    break
            if king_pos:
                break
        if not king_pos:
            return True
        opponent = -player
        self.current_player = opponent
        legal_moves = self.get_legal_moves()
        self.current_player = player
        for move in legal_moves:
            if move[2:] == king_pos:
                log_info(f"King of player {player} threatened at {king_pos} by move {move}", verbose_only=True)
                return True
        return False

    def step(self, action):
        legal_moves = self.get_legal_moves()
        self.action_space = legal_moves

        if not legal_moves:
            if self.is_king_threatened(self.current_player):
                reward = -2 if self.current_player == self.computer_player else 2
                log_info(f"Game ended: Player {-self.current_player} wins, Reward={reward}")
                return self.get_state(), reward, True
            else:
                return self.get_state(), -0.1, True

        if action not in legal_moves:
            log_error(f"Invalid action {action} not in valid moves: {legal_moves[:5]}... (total {len(legal_moves)})")
            return self.get_state(), -1.0, False

        i, j, ni, nj = action
        moving_piece = self.board[i, j]
        captured_piece = self.board[ni, nj]

        self.board[ni, nj] = moving_piece
        self.board[i, j] = 0

        reward = -0.01
        done = False

        if captured_piece * self.current_player < 0:
            base_reward = self.piece_strategic_values.get(self.piece_values.get(abs(captured_piece), '兵'), 1) / 10
            reward += min(base_reward, 0.5)
            log_info(f"Captured piece {self.piece_values.get(abs(captured_piece), 'Unknown')}, Reward increased by {base_reward}")

        if abs(captured_piece) == 1:
            reward = 2 if self.current_player == self.computer_player else -2
            done = True
            log_info(f"King captured, Reward={reward}")

        if reward == -0.01:
            log_warning(f"No significant reward change at step, action={action}")

        reward = np.clip(reward, -2, 2)

        self.current_player = -self.current_player
        self.action_space = self.get_legal_moves()
        log_info(f"Step executed: Action={action}, Reward={reward}, Done={done}")
        return self.get_state(), reward, done

    def render_pygame(self):
        if not self.pygame_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 900))
            self.pygame_initialized = True
            log_info("Pygame initialized for rendering")
        self.screen.fill((255, 255, 255))
        for i in range(9):
            pygame.draw.line(self.screen, (0, 0, 0), (50, 50 + i * 80), (770, 50 + i * 80))
        for j in range(10):
            pygame.draw.line(self.screen, (0, 0, 0), (50 + j * 80, 50), (50 + j * 80, 770))
        pygame.draw.rect(self.screen, (135, 206, 235), (50, 370, 720, 80))
        font = pygame.font.SysFont('Microsoft YaHei', 40)
        text = font.render('楚河      汉界', True, (0, 0, 0))
        self.screen.blit(text, (300, 405))
        for i in range(9):
            for j in range(10):
                if self.board[i, j] != 0:
                    piece = self.piece_values.get(self.board[i, j], '未知')
                    text = font.render(piece, True, (255, 0, 0) if self.board[i, j] > 0 else (0, 0, 0))
                    self.screen.blit(text, (50 + j * 80 - 20, 50 + i * 80 - 20))
                    log_info(f"Rendered piece {piece} at position ({i}, {j})", verbose_only=True)
        pygame.display.flip()
        log_info("Rendered board with Chinese characters", verbose_only=True)

    def get_move_from_clicks(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise SystemExit
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    col = (x - 50) // 80
                    row = (y - 50) // 80
                    if 0 <= row < 9 and 0 <= col < 10:
                        return (row, col, row, col)  # 简化为单点击，返回初始位置
            return None

    def close(self):
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False
            log_info("Pygame closed")