from datetime import datetime

import numpy as np
import pygame
import logging
import time
from collections import defaultdict

# Configure logging
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


def strict_dim_check(func):
    def wrapper(env, *args, **kwargs):
        result = func(env, *args, **kwargs)
        if not hasattr(env, 'state_dim'):
            return result

        actual_dim = result.shape[0] if hasattr(result, 'shape') else len(result)
        if actual_dim != env.state_dim:
            raise ValueError(
                f"致命维度冲突！\n"
                f"环境声明维度: {env.state_dim}\n"
                f"实际生成维度: {actual_dim}\n"
                f"冲突方法: {func.__name__}"
            )
        return result

    return wrapper

class ChineseChessEnv:
    def __init__(self, computer_player=1, agent=None):
        self.state_dim = 1530  # 原错误设置为90，必须改为1530
        self.board = np.zeros((10, 9))
        self.current_player = 1
        self.computer_player = computer_player
        self.agent = agent
        self.action_space = []
        self.pygame_initialized = False
        self.start_time = time.time()
        self.captured_pieces = {'red': [], 'black': []}
        self.move_count_since_capture = 0  # For draw by 80-move rule
        self.check_status = None  # Track if current player is in check
        self.selected_piece = None  # Track selected piece
        self.valid_moves = []  # Track valid moves for selected piece

        self.piece_values = {
            1: '帅', 2: '仕', 3: '相', 4: '马', 5: '车', 6: '炮', 7: '兵',
            -1: '将', -2: '士', -3: '象', -4: '马', -5: '车', -6: '炮', -7: '卒'
        }
        self.piece_strategic_values = {
            '帅': 1000, '将': 1000,  # Increased king value
            '仕': 20, '士': 20,
            '相': 20, '象': 20,
            '马': 40,
            '车': 90,
            '炮': 45,
            '兵': 10, '卒': 10
        }
        log_info(f"Environment initialized with computer_player={computer_player}")

    @strict_dim_check
    def reset(self):
        self.board = np.zeros((10, 9))
        self.start_time = time.time()
        self.captured_pieces = {'red': [], 'black': []}
        self.move_count_since_capture = 0
        self.check_status = None
        self.selected_piece = None
        self.valid_moves = []

        # Black pieces (top side)
        self.board[0, :] = [-5, -4, -3, -2, -1, -2, -3, -4, -5]
        self.board[2, 1] = -6
        self.board[2, 7] = -6
        self.board[3, 0::2] = -7

        # Red pieces (bottom side)
        self.board[9, :] = [5, 4, 3, 2, 1, 2, 3, 4, 5]
        self.board[7, 1] = 6
        self.board[7, 7] = 6
        self.board[6, 0::2] = 7

        self.current_player = 1
        self.action_space = self.get_legal_moves()
        state = self.get_state()
        # print(f"重置后状态维度: {len(state)}")  # 应为1530
        return state

    @strict_dim_check
    def get_state(self):
        """返回严格1530维的状态表示"""
        # 基础棋盘层 (1 channel)
        state_layers = [self.board.copy()]

        # 棋子类型通道 (14 channels)
        for piece_type in [1, 2, 3, 4, 5, 6, 7]:
            state_layers.append((self.board == piece_type).astype(float))
            state_layers.append((self.board == -piece_type).astype(float))

        # 历史信息通道 (2 channels)
        state_layers.extend([
            (self.board * self.current_player) > 0,  # 当前方控制
            (self.board * self.current_player) < 0  # 对手控制
        ])

        state = np.concatenate([layer.flatten() for layer in state_layers])
        assert len(state) == self.state_dim, "维度必须一致！"
        return state

    def get_legal_moves(self):
        """改进的合法移动检查"""
        legal_moves = []
        must_respond_moves = []  # 必须应对将军的移动

        # 检查是否被将军
        in_check = self.is_king_threatened(self.current_player)

        for i in range(10):
            for j in range(9):
                piece = self.board[i, j]
                if piece * self.current_player <= 0:  # 不是当前玩家的棋子
                    continue

                moves = self._get_piece_moves(i, j, abs(piece))

                for ni, nj in moves:
                    # 模拟移动
                    temp_board = self.board.copy()
                    temp_board[ni, nj] = temp_board[i, j]
                    temp_board[i, j] = 0

                    # 检查移动后是否仍被将军
                    if not self._simulate_check(temp_board, self.current_player):
                        if in_check:
                            must_respond_moves.append((i, j, ni, nj))
                        else:
                            legal_moves.append((i, j, ni, nj))

        # 如果被将军，只返回能解除将军的移动
        if in_check:
            return must_respond_moves if must_respond_moves else []

        return legal_moves if legal_moves else []

    def _get_piece_moves(self, i, j, piece_type):
        """严格修正后的棋子移动生成函数"""
        moves = []
        current_player_sign = self.current_player

        def add_move(ni, nj):
            """安全添加移动位置（自动边界检查）"""
            if 0 <= ni < 10 and 0 <= nj < 9:
                moves.append((ni, nj))

        # 将/帅 (King)
        if piece_type == 1:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                # 红方九宫(7-9行) 黑方九宫(0-2行)
                if ((current_player_sign == 1 and 7 <= ni <= 9) or
                    (current_player_sign == -1 and 0 <= ni <= 2)) and \
                        3 <= nj <= 5:  # 中线3-5列
                    add_move(ni, nj)

        # 士/仕 (Advisor)
        elif piece_type == 2:
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = i + dx, j + dy
                # 必须严格在九宫格内
                if ((current_player_sign == 1 and 7 <= ni <= 9 and 3 <= nj <= 5) or
                        (current_player_sign == -1 and 0 <= ni <= 2 and 3 <= nj <= 5)):
                    add_move(ni, nj)

        # 象/相 (Elephant) - 完全重写
        elif piece_type == 3:
            for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                ni, nj = i + dx, j + dy
                # 象眼位置(田字中心)
                block_i, block_j = i + dx // 2, j + dy // 2

                # 检查：1.不越界 2.不过河 3.象眼无子
                if (0 <= ni < 10 and 0 <= nj < 9 and
                        0 <= block_i < 10 and 0 <= block_j < 9 and
                        self.board[block_i, block_j] == 0):

                    # 红方不过河(5-9行) 黑方不过河(0-4行)
                    if (current_player_sign == 1 and ni >= 5) or \
                            (current_player_sign == -1 and ni <= 4):
                        add_move(ni, nj)

        # 马 (Horse)
        elif piece_type == 4:
            horse_legs = [
                ((-2, -1), (-1, 0)), ((-2, 1), (-1, 0)),
                ((2, -1), (1, 0)), ((2, 1), (1, 0)),
                ((-1, -2), (0, -1)), ((-1, 2), (0, 1)),
                ((1, -2), (0, -1)), ((1, 2), (0, 1))
            ]
            for (dx, dy), (bx, by) in horse_legs:
                ni, nj = i + dx, j + dy
                bi, bj = i + bx, j + by
                if (0 <= ni < 10 and 0 <= nj < 9 and
                        0 <= bi < 10 and 0 <= bj < 9 and
                        self.board[bi, bj] == 0):  # 马腿无子
                    add_move(ni, nj)

        # 车 (Rook)
        elif piece_type == 5:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                while 0 <= ni < 10 and 0 <= nj < 9:
                    target = self.board[ni, nj]
                    if target == 0:
                        add_move(ni, nj)
                    else:
                        if target * current_player_sign < 0:  # 可吃对方子
                            add_move(ni, nj)
                        break
                    ni += dx
                    nj += dy

        # 炮 (Cannon)
        elif piece_type == 6:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                has_jumped = False
                while 0 <= ni < 10 and 0 <= nj < 9:
                    target = self.board[ni, nj]
                    if not has_jumped:
                        if target == 0:
                            add_move(ni, nj)
                        else:
                            has_jumped = True
                    else:
                        if target != 0:
                            if target * current_player_sign < 0:  # 可吃对方子
                                add_move(ni, nj)
                            break
                    ni += dx
                    nj += dy

        # 兵/卒 (Pawn)
        elif piece_type == 7:
            if current_player_sign == 1:  # 红兵
                if i > 0:  # 可前进
                    add_move(i - 1, j)
                if i <= 4:  # 已过河(0-4行)
                    if j > 0: add_move(i, j - 1)  # 左移
                    if j < 8: add_move(i, j + 1)  # 右移
            else:  # 黑卒
                if i < 9:  # 可前进
                    add_move(i + 1, j)
                if i >= 5:  # 已过河(5-9行)
                    if j > 0: add_move(i, j - 1)
                    if j < 8: add_move(i, j + 1)

        # 最终过滤：排除吃己方棋子的移动
        return [(ni, nj) for (ni, nj) in moves
                if self.board[ni, nj] * current_player_sign <= 0]

    def _simulate_check(self, board, player):
        # Get king position
        king_pos = None
        for i in range(10):
            for j in range(9):
                if board[i, j] == player:
                    king_pos = (i, j)
                    break
            if king_pos:
                break
        if not king_pos:
            return True  # king already gone

        opponent = -player

        # Check all opponent pieces that could attack the king
        for i in range(10):
            for j in range(9):
                if board[i, j] * opponent > 0:
                    piece = abs(board[i, j])
                    if piece == 5:  # Rook
                        if self._check_rook_attack(board, i, j, king_pos[0], king_pos[1], opponent):
                            return True
                    elif piece == 6:  # Cannon
                        if self._check_cannon_attack(board, i, j, king_pos[0], king_pos[1], opponent):
                            return True
                    elif piece == 4:  # Horse
                        if self._check_horse_attack(board, i, j, king_pos[0], king_pos[1], opponent):
                            return True
                    elif piece == 1:  # King (facing kings)F
                        if j == king_pos[1]:  # Same file
                            if self._check_kings_facing(board):
                                return True
                    elif piece == 7:
                        if self._check_pawn_attack( i, j, king_pos[0], king_pos[1], opponent):
                            return True
        return False

    @staticmethod
    def _check_rook_attack(board, i, j, ki, kj, opponent):
        """Check if rook at (i,j) can attack king at (极,kj)"""
        if i == ki:  # Same row
            step = 1 if kj > j else -1
            for nj in range(j + step, kj, step):
                if board[i, nj] != 0:
                    return False
            return True
        elif j == kj:  # Same column
            step = 1 if ki > i else -1
            for ni in range(i + step, ki, step):
                if board[ni, j] != 0:
                    return False
            return True
        return False

    def _check_cannon_attack(self, board, i, j, ki, kj, opponent):
        """Check if cannon at (i,j) can attack king at (ki,kj)"""
        if i == ki:  # Same row
            step = 1 if kj > j else -1
            jumped = False
            for nj in range(j + step, kj, step):
                if board[i, nj] != 0:
                    if not jumped:
                        jumped = True
                    else:
                        return False
            return jumped  # Must jump exactly one piece
        elif j == kj:  # Same column
            step = 1 if ki > i else -1
            jumped = False
            for ni in range(i + step, ki, step):
                if board[ni, j] != 0:
                    if not jumped:
                        jumped = True
                    else:
                        return False
            return jumped  # Must jump exactly one piece
        return False

    def _check_horse_attack(self, board, i, j, ki, kj, opponent):
        """Check if horse at (i,j) can attack king at (ki,kj)"""
        for dx, dy, bx, by in [
            (-2, -1, -1, 0), (-2, 1, -1, 0),
            (2, -1, 1, 0), (2, 1, 1, 0),
            (-1, -2, 0, -1), (-1, 2, 0, 1),
            (1, -2, 0, -1), (1, 2, 0, 1)
        ]:
            ni, nj = i + dx, j + dy
            bi, bj = i + bx, j + by
            if ni == ki and nj == kj:
                if 0 <= bi < 10 and 0 <= bj < 9:
                    if board[bi, bj] == 0:  # Horse leg not blocked
                        return True
        return False

    def _check_kings_facing(self, board):
        """Check if kings are facing each other with no pieces in between"""
        red_king_pos = None
        black_king_pos = None
        for i in range(10):
            for j in range(9):
                if board[i, j] == 1:  # Red king
                    red_king_pos = (i, j)
                elif board[i, j] == -1:  # Black king
                    black_king_pos = (i, j)

        if not red_king_pos or not black_king_pos:
            return False

        # Check if on same file
        if red_king_pos[1] == black_king_pos[1]:
            # Check if no pieces between them
            start = min(red_king_pos[0], black_king_pos[0]) + 1
            end = max(red_king_pos[0], black_king_pos[0])
            for i in range(start, end):
                if board[i, red_king_pos[1]] != 0:
                    return False
            return True
        return False

    def _check_pawn_attack(self,pi, pj, ki, kj, opponent):
        """
        兵卒将军检测
        参数:
            pi,pj: 兵卒位置
            ki,kj: 将/帅位置
            opponent: 当前对手（兵卒方）
        """
        # 兵卒只能近距离将军（相邻四方向）
        if abs(pi - ki) + abs(pj - kj) != 1:
            return False

        # 红兵(7)只能向下攻击（数值越小越靠近黑方）
        if opponent == -1:  # 红兵攻击黑将
            return ki >= pi  # 将的位置在兵的下方

        # 黑卒(-7)只能向上攻击（数值越大越靠近红方）
        else:  # 黑卒攻击红帅
            return ki <= pi  # 帅的位置在卒的上方

    def step(self, action):
        """
        修复版step函数，确保：
        1. 正确记录吃子信息
        2. 更新将军状态
        3. 保持原有核心功能
        """
        # ===== 1. 初始状态验证 =====
        if not self._is_validate_board():
            return self.get_state(), -1.0, True, {"error": "invalid_board"}

        # ===== 2. 获取合法移动 =====
        legal_moves = self.get_legal_moves()

        # ===== 3. 终局判定 =====
        # 情况1: 无合法移动
        if not legal_moves:
            if self._simulate_check(self.board, self.current_player):
                # 被将死
                reward = -1.0 if self.current_player == self.computer_player else 1.0
                return self.get_state(), reward, True, "checkmate"
            else:
                # 困毙（当前玩家负）
                reward = -1.0 if self.current_player == self.computer_player else 1.0
                return self.get_state(), reward, True, "stalemate"

        # 情况2: 80回合未吃子（和棋）
        if self.move_count_since_capture >= 80:
            return self.get_state(), 0.0, True, "draw_80_moves"

        # ===== 4. 动作验证 =====
        if action not in legal_moves:
            return self.get_state(), -0.1, False, {"error": "invalid_move"}

        # ===== 5. 执行移动 =====
        i, j, ni, nj = action
        moving_piece = self.board[i, j]
        captured_piece = self.board[ni, nj]

        # 保存旧状态用于回滚
        old_board = self.board.copy()
        old_player = self.current_player

        # 执行移动
        self.board[ni, nj] = moving_piece
        self.board[i, j] = 0

        # +++ 新增：记录吃子 +++
        if captured_piece != 0:
            piece_name = self.piece_values.get(abs(captured_piece), '?')
            if old_player == 1:  # 红方吃子
                self.captured_pieces['red'].append(piece_name)
            else:  # 黑方吃子
                self.captured_pieces['black'].append(piece_name)
            self.move_count_since_capture = 0
        else:
            self.move_count_since_capture += 1
        # --- 新增结束 ---

        # ===== 6. 移动后验证 =====
        # 检查移动后是否自将
        if self._simulate_check(self.board, old_player):
            self.board = old_board  # 回滚
            return self.get_state(), -0.5, False, {"error": "self_check"}

        # 检查将帅对面
        if self._check_kings_facing(self.board):
            self.board = old_board
            return self.get_state(), 0.0, True, "draw_kings_facing"

        # +++ 新增：更新将军状态 +++
        self.check_status = None
        if self.is_king_threatened(-self.current_player):
            self.check_status = "red" if self.current_player == -1 else "black"
        # --- 新增结束 ---

        # ===== 7. 切换玩家 =====
        self.current_player *= -1

        # ===== 8. 简化奖励计算 =====
        reward = self._calculate_reward(moving_piece, captured_piece)

        return self.get_state(), reward, False, {
            "move": action,
            "captured": captured_piece if captured_piece != 0 else None,
            "check": self.check_status
        }

    def _calculate_reward(self, moving_piece, captured):
        """
        简化版奖励计算：
        1. 仅保留吃子奖励
        2. 移除所有需要prev_board的计算
        """
        reward = 0.0

        # 1. 吃子奖励（基于棋子价值）
        if captured != 0:
            piece_type = self.piece_values[abs(captured)]
            value = self.piece_strategic_values[piece_type]
            reward += np.tanh(value * 0.01)  # 压缩到(-1,1)

        # 2. 将军奖励（不需要历史状态）
        if self.is_king_threatened(-self.current_player):
            reward += 0.05 if self.current_player == self.computer_player else -0.05

        return np.clip(reward, -1.0, 1.0)

    def _evaluate_board(self, board):
        """
        保留局面评估函数（其他功能可能使用）
        但不再用于奖励计算
        """
        return np.sum([
            self.piece_strategic_values[self.piece_values[abs(p)]]
            for p in board.flatten() if p != 0
        ])

    def _undo_move(self, i, j, ni, nj, captured):
        """回滚移动"""
        self.board[i, j] = self.board[ni, nj]
        self.board[ni, nj] = captured

    def is_king_threatened(self, player):
        return self._simulate_check(self.board, player)

    def evaluate_position(self):
        """Evaluate the current board position from red's perspective"""
        score = 0
        for i in range(10):
            for j in range(9):
                piece = self.board[i, j]
                if piece != 0:
                    piece_name = self.piece_values.get(abs(piece), '')
                    value = self.piece_strategic_values.get(piece_name, 0)
                    score += value if piece > 0 else -value
        return score

    def render_pygame(self):
        if not self.pygame_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((1000, 850))  # Increased height to 850
            pygame.display.set_caption("中国象棋")
            self.pygame_initialized = True

        # Wooden background
        self.screen.fill((238, 215, 167))

        # Draw info panel on left
        self._draw_info_panel()

        # Draw chess board (offset to right)
        self._draw_chess_board(200)  # 200px offset

        # Draw highlights and move hints if any
        if self.selected_piece:
            self._highlight_position(*self.selected_piece)
            self._draw_move_hint(self.selected_piece)

        pygame.display.flip()

    def _draw_info_panel(self):
        """Draw the information panel on the left side"""
        panel_width = 180
        pygame.draw.rect(self.screen, (200, 180, 150), (10, 10, panel_width, 830))

        font = pygame.font.SysFont('SimHei', 24)

        # Current turn
        turn_text = "红方回合" if self.current_player == 1 else "黑方回合"
        text_surface = font.render(turn_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (20, 20))

        # Evaluation
        score = self.evaluate_position()
        score_text = f"局面评估: {score:.1f}"
        score_surface = font.render(score_text, True, (0, 0, 0))
        self.screen.blit(score_surface, (20, 60))

        # Timer
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        timer_text = f"用时: {mins:02d}:{secs:02d}"
        timer_surface = font.render(timer_text, True, (0, 0, 0))
        self.screen.blit(timer_surface, (20, 100))

        # +++ 修复：将军状态显示 +++
        if self.check_status:
            if self.check_status == "red":
                check_text = "红方被将军!"
                check_color = (200, 0, 0)
            else:
                check_text = "黑方被将军!"
                check_color = (0, 0, 0)
            check_surface = font.render(check_text, True, check_color)
            self.screen.blit(check_surface, (20, 140))
        # --- 修复结束 ---

        # +++ 修复：吃子列表显示位置 +++
        y_start = 180 if self.check_status else 140
        self._draw_captured_pieces(20, y_start)

        # Move count
        move_text = f"回合: {self.move_count_since_capture}/80"
        move_surface = font.render(move_text, True, (0, 0, 0))
        self.screen.blit(move_surface, (20, 800))

    def _draw_captured_pieces(self, x, y):
        """Draw the captured pieces lists with improved layout"""
        font = pygame.font.SysFont('SimHei', 24)

        # 红方吃子（黑方棋子）
        text = font.render("红方吃子:", True, (200, 0, 0))  # 红色标题
        self.screen.blit(text, (x, y))

        piece_font = pygame.font.SysFont('SimHei', 20)
        captured_red = self.captured_pieces['red'][-10:]  # 显示最后10个

        # 计算行数
        rows = (len(captured_red) + 2) // 3

        for i, piece in enumerate(captured_red):
            text = piece_font.render(piece, True, (0, 0, 0))  # 黑色棋子
            self.screen.blit(text, (x + 20 + (i % 3) * 50, y + 30 + (i // 3) * 25))

        # 黑方吃子（红方棋子）
        y += 30 + rows * 25  # 动态间距
        text = font.render("黑方吃子:", True, (0, 0, 0))  # 黑色标题
        self.screen.blit(text, (x, y))

        captured_black = self.captured_pieces['black'][-10:]
        rows = (len(captured_black) + 2) // 3

        for i, piece in enumerate(captured_black):
            text = piece_font.render(piece, True, (200, 0, 0))  # 红色棋子
            self.screen.blit(text, (x + 20 + (i % 3) * 50, y + 30 + (i // 3) * 25))

    def _draw_chess_board(self, x_offset):
        """Draw the chess board at the given x offset"""
        cell_size = 80
        y_offset = 25  # 较小的垂直偏移，确保黑方棋子完全显示

        # Draw all horizontal lines
        for i in range(11):  # 10 horizontal lines (11 rows)
            y_pos = y_offset + i * cell_size
            pygame.draw.line(self.screen, (0, 0, 0),
                             (x_offset, y_pos),
                             (x_offset + 8 * cell_size, y_pos), 2)

        # Draw vertical lines - special handling for river area
        for j in range(9):  # 9 vertical lines (8 columns)
            x_pos = x_offset + j * cell_size
            # Only draw left and right boundary lines in river area (between rows 4 and 5)
            if j == 0 or j == 8:  # Boundary lines
                pygame.draw.line(self.screen, (0, 0, 0),
                                 (x_pos, y_offset),
                                 (x_pos, y_offset + 9 * cell_size), 2)
            else:  # Non-boundary lines
                # Above river (rows 0-4)
                pygame.draw.line(self.screen, (0, 0, 0),
                                 (x_pos, y_offset),
                                 (x_pos, y_offset + 4 * cell_size), 2)
                # Below river (rows 5-9)
                pygame.draw.line(self.screen, (0, 0, 0),
                                 (x_pos, y_offset + 5 * cell_size),
                                 (x_pos, y_offset + 9 * cell_size), 2)

        # Draw palaces
        # Upper palace
        pygame.draw.line(self.screen, (0, 0, 0),
                         (x_offset + 3 * cell_size, y_offset),
                         (x_offset + 5 * cell_size, y_offset + 2 * cell_size), 2)
        pygame.draw.line(self.screen, (0, 0, 0),
                         (x_offset + 5 * cell_size, y_offset),
                         (x_offset + 3 * cell_size, y_offset + 2 * cell_size), 2)

        # Lower palace
        pygame.draw.line(self.screen, (0, 0, 0),
                         (x_offset + 3 * cell_size, y_offset + 7 * cell_size),
                         (x_offset + 5 * cell_size, y_offset + 9 * cell_size), 2)
        pygame.draw.line(self.screen, (0, 0, 0),
                         (x_offset + 5 * cell_size, y_offset + 7 * cell_size),
                         (x_offset + 3 * cell_size, y_offset + 9 * cell_size), 2)

        # River
        river_font = pygame.font.SysFont('SimHei', 40)
        text = river_font.render('楚河  汉界', True, (0, 0, 0))
        text_rect = text.get_rect(center=(x_offset + 4 * cell_size, y_offset + 4.5 * cell_size))
        self.screen.blit(text, text_rect)

        # Draw pieces
        piece_font = pygame.font.SysFont('SimHei', 36)
        for i in range(10):
            for j in range(9):
                piece = self.board[i, j]
                if piece != 0:
                    x = x_offset + j * cell_size
                    y = y_offset + i * cell_size

                    # Piece background
                    pygame.draw.circle(self.screen,
                                       (255, 230, 230) if piece > 0 else (100, 50, 0),
                                       (x, y), 35)  # Increased size
                    pygame.draw.circle(self.screen, (0, 0, 0), (x, y), 35, 2)

                    # Piece character
                    label = self.piece_values.get(piece, '?')
                    color = (200, 0, 0) if piece > 0 else (255, 255, 255)
                    piece_text = piece_font.render(label, True, color)
                    text_rect = piece_text.get_rect(center=(x, y))
                    self.screen.blit(piece_text, text_rect)

    def close(self):
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False
            log_info("Pygame closed")

    def get_move_from_clicks(self):
        if not self.pygame_initialized:
            self.render_pygame()

        move = None
        clock = pygame.time.Clock()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if x < 200:  # Clicked on info panel
                        continue

                    # Convert coordinates
                    col = (x - 200) // 80
                    row = (y - 25) // 80  # Account for vertical offset

                    if 0 <= col < 9 and 0 <= row < 10:
                        # First click: select a piece
                        if self.selected_piece is None:
                            # Check if clicked on own piece
                            if self.board[row, col] * self.current_player > 0:
                                self.selected_piece = (row, col)
                                # Get valid moves for this piece
                                piece_type = abs(self.board[row, col])
                                self.valid_moves = self._get_piece_moves(row, col, piece_type)
                                # Filter moves that are actually valid
                                self.valid_moves = [
                                    (ni, nj) for ni, nj in self.valid_moves
                                    if self.board[ni, nj] * self.current_player <= 0
                                ]
                                log_info(f"Selected piece at ({row},{col}) with {len(self.valid_moves)} valid moves")

                        # Second click: move the piece
                        else:
                            if (row, col) in self.valid_moves:
                                move = (*self.selected_piece, row, col)
                                done = True
                            else:
                                # Clicked on another piece - select it instead
                                if self.board[row, col] * self.current_player > 0:
                                    self.selected_piece = (row, col)
                                    # Get valid moves for this piece
                                    piece_type = abs(self.board[row, col])
                                    self.valid_moves = self._get_piece_moves(row, col, piece_type)
                                    # Filter moves that are actually valid
                                    self.valid_moves = [
                                        (ni, nj) for ni, nj in self.valid_moves
                                        if self.board[ni, nj] * self.current_player <= 0
                                    ]
                                    log_info(
                                        f"Selected new piece at ({row},{col}) with {len(self.valid_moves)} valid moves")
                                else:
                                    # Invalid move, clear selection
                                    self.selected_piece = None
                                    self.valid_moves = []
                                    log_info("Invalid move, clearing selection")

            # Render the board with highlights
            self.render_pygame()
            pygame.display.flip()
            clock.tick(30)

        # Clear selection after move
        self.selected_piece = None
        self.valid_moves = []
        return move

    def _highlight_position(self, row, col):
        """Highlight the selected piece position"""
        x = 200 + col * 80
        y = 25 + row * 80  # Account for vertical offset
        pygame.draw.rect(self.screen, (0, 255, 0), (x - 40, y - 40, 80, 80), 4)

    def _draw_move_hint(self, selected_piece):
        """Draw possible move positions for the selected piece"""
        row, col = selected_piece

        for ni, nj in self.valid_moves:
            x = 200 + nj * 80
            y = 25 + ni * 80  # Account for vertical offset

            # Draw green circle for empty spots, red for captures
            if self.board[ni, nj] == 0:
                pygame.draw.circle(self.screen, (100, 255, 100), (x, y), 15)
            else:
                pygame.draw.circle(self.screen, (255, 100, 100), (x, y), 20)

    def _is_validate_board(self):
        """修正版棋盘验证（解决将帅位置颠倒问题）"""
        # 1. 将帅存在性检查
        red_kings = np.sum(self.board == 1)  # 红帅
        black_kings = np.sum(self.board == -1)  # 黑将

        if red_kings != 1 or black_kings != 1:
            print(f"将帅数量异常：红帅={red_kings} 黑将={black_kings}")
            self._log_board_state()
            return False

        # 2. 获取真实位置（注意红方在棋盘下方）
        try:
            red_pos = np.argwhere(self.board == 1)[0]  # 红帅位置（应7-9行）
            black_pos = np.argwhere(self.board == -1)[0]  # 黑将位置（应0-2行）
        except IndexError:
            print("将帅位置获取失败")
            return False

        # 3. 正确的位置验证（根据实际棋盘布局）
        # 红帅应在下方九宫格（7-9行，3-5列）
        red_valid = (7 <= red_pos[0] <= 9) and (3 <= red_pos[1] <= 5)
        # 黑将应在上方九宫格（0-2行，3-5列）
        black_valid = (0 <= black_pos[0] <= 2) and (3 <= black_pos[1] <= 5)

        if not red_valid:
            print(f"红帅位置非法：row={red_pos[0]} col={red_pos[1]}")
        if not black_valid:
            print(f"黑将位置非法：row={black_pos[0]} col={black_pos[1]}")

        return red_valid and black_valid

    def _log_board_state(self):
        """记录错误棋盘状态"""
        with open("errors/board_errors.log", "a") as f:
            f.write(f"\n异常时间：{datetime.now()}\n")
            f.write("当前棋盘：\n")
            f.write(str(self.board) + "\n")
            f.write(f"当前玩家：{'红方' if self.current_player == 1 else '黑方'}\n")