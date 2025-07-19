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
        self.action_space = []  # Initialize as empty, update dynamically
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
        self.board[9 - 1, 0] = 5
        self.board[9 - 1, 1] = 4
        self.board[9 - 1, 2] = 3
        self.board[9 - 1, 3] = 2
        self.board[9 - 1, 4] = 1
        self.board[9 - 1, 5] = 2
        self.board[9 - 1, 6] = 3
        self.board[9 - 1, 7] = 4
        self.board[9 - 1, 8] = 5
        self.board[9 - 3, 1] = 6
        self.board[9 - 3, 7] = 6
        self.board[9 - 4, 0] = 7
        self.board[9 - 4, 2] = 7
        self.board[9 - 4, 4] = 7
        self.board[9 - 4, 6] = 7
        self.board[9 - 4, 8] = 7
        # Black pieces
        self.board[0, 0] = -5
        self.board[0, 1] = -4
        self.board[0, 2] = -3
        self.board[0, 3] = -2
        self.board[0, 4] = -1
        self.board[0, 5] = -2
        self.board[0, 6] = -3
        self.board[0, 7] = -4
        self.board[0, 8] = -5
        self.board[2, 1] = -6
        self.board[2, 7] = -6
        self.board[3, 0] = -7
        self.board[3, 2] = -7
        self.board[3, 4] = -7
        self.board[3, 6] = -7
        self.board[3, 8] = -7

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
                    legal_moves.append((i, j, i, j))
        return legal_moves

    def step(self, action):
        legal_moves = self.get_legal_moves()
        self.action_space = legal_moves

        if not legal_moves:
            return self.get_state(), -0.1, True

        if action not in legal_moves:
            log_error(f"Invalid action {action} not in valid moves: {legal_moves[:5]}... (total {len(legal_moves)})")
            return self.get_state(), -1.0, False

        i, j, ni, nj = action
        self.board[ni, nj] = self.board[i, j]
        self.board[i, j] = 0

        reward = -0.01
        done = False
        if abs(self.board[ni, nj]) == 1:
            reward = 2 if self.current_player == self.computer_player else -2
            done = True

        self.current_player = -self.current_player
        self.action_space = self.get_legal_moves()
        return self.get_state(), reward, done

    def render_pygame(self):
        if not self.pygame_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 900))
            self.pygame_initialized = True
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
        pygame.display.flip()

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
                        return (row, col, row, col)
            return None

    def close(self):
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False
            log_info("Pygame closed")