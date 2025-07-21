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


class ChineseChessEnv:
    def __init__(self, computer_player=1, agent=None):
        self.board = np.zeros((10, 9))  # 10横线与9纵线的交点 (0,0)到(9,8)
        self.current_player = 1
        self.computer_player = computer_player
        self.agent = agent
        self.state_dim = 90  # 初始状态维度，10*9=90
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
        return self.get_state()

    def get_state(self):
        return self.board.flatten()

    def get_legal_moves(self):
        legal_moves = []
        must_respond_moves = []  # Moves that respond to check

        in_check = self.is_king_threatened(self.current_player)
        # 修复将军显示错误
        if in_check:
            self.check_status = "red" if self.current_player == -1 else "black"
        else:
            self.check_status = None

        for i in range(10):
            for j in range(9):
                piece = self.board[i, j]
                if piece * self.current_player <= 0:
                    continue

                abs_piece = abs(piece)
                moves = self._get_piece_moves(i, j, abs_piece)

                for ni, nj in moves:
                    # Simulate move to check validity
                    temp_board = self.board.copy()
                    temp_board[ni, nj] = temp_board[i, j]
                    temp_board[i, j] = 0

                    # Check if move leaves king in check or creates facing kings
                    if not self._simulate_check(temp_board, self.current_player) and not self._check_kings_facing(
                            temp_board):
                        if in_check:
                            must_respond_moves.append((i, j, ni, nj))
                        else:
                            if self.board[ni, nj] * self.current_player <= 0:
                                legal_moves.append((i, j, ni, nj))

        # If in check, only return moves that get out of check
        if in_check:
            return must_respond_moves if must_respond_moves else []

        return legal_moves if legal_moves else []

    def _get_piece_moves(self, i, j, piece_type):
        """Helper function to get possible moves for a piece (without checking board state)"""
        moves = []

        if piece_type == 1:  # King/General
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < 10 and 0 <= nj < 9 and 3 <= nj <= 5:
                    if (self.current_player == 1 and 7 <= ni <= 9) or (self.current_player == -1 and 0 <= ni <= 2):
                        moves.append((ni, nj))

        elif piece_type == 2:  # Advisor
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < 10 and 0 <= nj < 9 and 3 <= nj <= 5:
                    if (self.current_player == 1 and 7 <= ni <= 9) or (self.current_player == -1 and 0 <= ni <= 2):
                        moves.append((ni, nj))

        elif piece_type == 3:  # Elephant
            for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                ni, nj = i + dx, j + dy
                mi, mj = i + dx // 2, j + dy // 2  # Center position
                if (0 <= ni < 10 and 0 <= nj < 9 and
                        0 <= mi < 10 and 0 <= mj < 9):
                    # Check if elephant center is blocked
                    if self.board[mi, mj] == 0:
                        if (self.current_player == 1 and ni >= 5) or (self.current_player == -1 and ni <= 4):
                            moves.append((ni, nj))

        elif piece_type == 4:  # Horse
            for dx, dy, bx, by in [
                (-2, -1, -1, 0), (-2, 1, -1, 0),
                (2, -1, 1, 0), (2, 1, 1, 0),
                (-1, -2, 0, -1), (-1, 2, 0, 1),
                (1, -2, 0, -1), (1, 2, 0, 1)
            ]:
                ni, nj = i + dx, j + dy
                bi, bj = i + bx, j + by  # Blocking position
                if (0 <= ni < 10 and 0 <= nj < 9 and
                        0 <= bi < 10 and 0 <= bj < 9):
                    # Check if horse leg is blocked
                    if self.board[bi, bj] == 0:
                        moves.append((ni, nj))

        elif piece_type == 5:  # Rook
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                while 0 <= ni < 10 and 0 <= nj < 9:
                    if self.board[ni, nj] == 0:
                        moves.append((ni, nj))
                    else:
                        if self.board[ni, nj] * self.current_player < 0:
                            moves.append((ni, nj))
                        break
                    ni += dx
                    nj += dy

        elif piece_type == 6:  # Cannon
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                jumped = False
                while 0 <= ni < 10 and 0 <= nj < 9:
                    if not jumped:
                        if self.board[ni, nj] == 0:
                            moves.append((ni, nj))
                        else:
                            jumped = True
                    else:
                        if self.board[ni, nj] != 0:
                            if self.board[ni, nj] * self.current_player < 0:
                                moves.append((ni, nj))
                            break
                    ni += dx
                    nj += dy

        elif piece_type == 7:  # Pawn
            if self.current_player == 1:  # Red
                if i > 0:  # Can move forward
                    moves.append((i - 1, j))
                if i <= 4:  # Crossed river - can move sideways
                    if j > 0:
                        moves.append((i, j - 1))
                    if j < 8:
                        moves.append((i, j + 1))
            else:  # Black
                if i < 9:  # Can move forward
                    moves.append((i + 1, j))
                if i >= 5:  # Crossed river - can move sideways
                    if j > 0:
                        moves.append((i, j - 1))
                    if j < 8:
                        moves.append((i, j + 1))

        return moves

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
                    elif piece == 1:  # King (facing kings)
                        if j == king_pos[1]:  # Same file
                            if self._check_kings_facing(board):
                                return True
                    # Other pieces can't directly check the king
        return False

    def _check_rook_attack(self, board, i, j, ki, kj, opponent):
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

    def step(self, action):
        legal_moves = self.get_legal_moves()
        if not legal_moves:
            if self.is_king_threatened(self.current_player):
                reward = -100 if self.current_player == self.computer_player else 100
                return self.get_state(), reward, True, "checkmate"
            return self.get_state(), -0.1, True, "stalemate"

        if action not in legal_moves:
            log_error(f"Invalid action {action} not in valid moves: {legal_moves[:5]}...")
            return self.get_state(), -1.0, False, "invalid"

        i, j, ni, nj = action
        captured = self.board[ni, nj]
        self.board[ni, nj] = self.board[i, j]
        self.board[i, j] = 0

        # Update move count since last capture
        if captured != 0:
            self.move_count_since_capture = 0
        else:
            self.move_count_since_capture += 1

        # Record captured pieces
        if captured * self.current_player < 0:
            side = 'red' if self.current_player == 1 else 'black'
            piece_name = self.piece_values.get(abs(captured), '?')
            self.captured_pieces[side].append(piece_name)

        reward = -0.01
        done = False
        status = "continue"

        # Check for draw by 80-move rule
        if self.move_count_since_capture >= 80:
            return self.get_state(), 0, True, "draw_80_moves"

        if captured * self.current_player < 0:
            captured_name = self.piece_values.get(abs(captured), '')
            reward += self.piece_strategic_values.get(captured_name, 0) / 100

        if abs(captured) == 1:  # Captured king
            reward = 100 if self.current_player == self.computer_player else -100
            done = True
            status = "checkmate"

        # Check if opponent is in check after move
        if self.is_king_threatened(-self.current_player):
            reward += 0.5 if self.current_player == self.computer_player else -0.5
            status = "check"

        reward = np.clip(reward, -100, 100)
        self.current_player *= -1
        return self.get_state(), reward, done, status

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

        # Check status - 修复将军显示错误
        if self.check_status:
            if self.check_status == "red":
                check_text = "黑方被将军!"
                check_color = (200, 0, 0)
            else:
                check_text = "红方被将军!"
                check_color = (0, 0, 0)
            check_surface = font.render(check_text, True, check_color)
            self.screen.blit(check_surface, (20, 140))

        # Captured pieces - improved layout
        self._draw_captured_pieces(20, 180 if self.check_status else 140)

        # Move count
        move_text = f"回合: {self.move_count_since_capture}/80"
        move_surface = font.render(move_text, True, (0, 0, 0))
        self.screen.blit(move_surface, (20, 800))  # Positioned at bottom left

    def _draw_captured_pieces(self, x, y):
        """Draw the captured pieces lists with improved layout"""
        font = pygame.font.SysFont('SimHei', 24)

        # Red captured pieces (by black)
        text = font.render("黑方吃子:", True, (0, 0, 0))
        self.screen.blit(text, (x, y))

        piece_font = pygame.font.SysFont('SimHei', 20)
        captured_black = self.captured_pieces['black'][-10:]  # Show last 10
        for i, piece in enumerate(captured_black):
            text = piece_font.render(piece, True, (200, 0, 0))
            self.screen.blit(text, (x + 20 + (i % 3) * 50, y + 30 + (i // 3) * 25))  # 3 columns

        # Black captured pieces (by red)
        y += 30 + ((len(captured_black) + 2) // 3) * 25  # Dynamic spacing based on items
        text = font.render("红方吃子:", True, (0, 0, 0))
        self.screen.blit(text, (x, y))

        captured_red = self.captured_pieces['red'][-10:]
        for i, piece in enumerate(captured_red):
            text = piece_font.render(piece, True, (0, 0, 0))
            self.screen.blit(text, (x + 20 + (i % 3) * 50, y + 30 + (i // 3) * 25))  # 3 columns

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
