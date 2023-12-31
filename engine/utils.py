import numpy as np

DEPTH_SEARCH = 3

# PIECE MOVEMENT (max 8 steps)
PIECE_NAMES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
MAX_STEPS = 8
diagonal_neighbours = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.int8)
straight_neighbours = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]], dtype=np.int8)
all_neighbours = np.vstack((diagonal_neighbours, straight_neighbours))

pawn_moves = straight_neighbours
pawn_captures = diagonal_neighbours
knight_moves = np.array(
    [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]],
    dtype=np.int8,
)
king_moves = all_neighbours

piece_values = np.array([1, 2, 5, 6, 10, 100], dtype=np.int8)

player_names = [
    "Blue",
    "Orange",
    "Green",
    "Red",
    "Purple",
    "Brown",
    "Pink",
    "Light Blue",
]


class Utils:
    def __init__(self, board_state):
        self.initial_board_state = board_state.copy()
        self.grid_height = board_state.shape[0]

        # Precalculate all possible moves
        self.all_moves = []
        for x in range(self.initial_board_state.shape[0]):
            x_moves = []
            for y in range(self.initial_board_state.shape[1]):
                position = np.array([x, y], dtype=np.int8)
                x_moves.append(
                    [self.precalculate_piece(position, piece) for piece in range(6)]
                )
            self.all_moves.append(x_moves)

    def precalculate_piece(self, position, piece):
        if piece == 0:
            pos = self.precalculate_moves(
                self.initial_board_state, position + pawn_moves
            )
            pos2 = self.precalculate_moves(
                self.initial_board_state, position + pawn_captures
            )
            return [pos, pos2]
        elif piece == 1:
            return self.precalculate_moves(
                self.initial_board_state, position + knight_moves
            )
        elif piece == 5:
            return self.precalculate_moves(
                self.initial_board_state, position + king_moves
            )
        else:
            return None

    def search_tree(
        self,
        board_state,
        player,
        distance=8,
        depth=DEPTH_SEARCH,
        alpha=-np.inf,
        beta=np.inf,
        is_player=True,
    ):
        is_root = depth == DEPTH_SEARCH

        # Get the best move for a player using the minimax algorithm with alpha-beta pruning
        if depth == 0:
            # If we've reached the maximum depth, evaluate the board state and return the score
            return self.evaluate_board(board_state, player)

        # Generate all possible moves for the current player
        moves = (
            self.get_moves_player(board_state, player)
            if is_player
            else self.get_moves_enemies(board_state, player, distance)
        )

        # Shuffle own (player, bot) moves
        if is_player:
            np.random.shuffle(moves)

        # If there are no possible moves, return the worst score
        if moves is None:
            return -np.inf if is_player else np.inf

        # Initialize the best score and move
        best_score = -np.inf if is_player else np.inf
        best_move = None

        # Loop over all possible moves and recursively search the game tree
        for move in moves:
            # Execute the move on a copy of the board state
            board_state_copy = self.execute_move(board_state, move, inplace=False)

            # Recursively search the game tree for the other player
            score = self.search_tree(
                board_state_copy,
                player,
                distance,
                depth=depth - 1,
                alpha=alpha,
                beta=beta,
                is_player=not is_player,
            )

            # Update the best score and move based on the current player
            if is_player:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
                if beta <= alpha:
                    break

        # Return the best score and move for the current player
        if is_root:
            return best_move
        else:
            return best_score

    def get_moves_player(self, board_state, player):
        own_mask, other_mask = self.get_masks(board_state, player)
        moves = []
        for piece in np.argwhere(own_mask):
            m = self.get_moves_position(board_state, piece, own_mask, other_mask)
            if m is not None and m.shape[0] > 0:
                # Add position to move
                m = np.hstack((np.tile(piece, (m.shape[0], 1)), m))
                moves.append(m)

        if moves:
            return np.vstack(moves)
        else:
            return np.array([])

    def get_moves_enemies(self, board_state, player, distance=-1):
        enemy_pieces, __ = self.get_enemy_pieces(board_state, player, distance)

        if enemy_pieces is None:
            return None

        # Also calc invalid moves (state == -2 or player_id)
        enemy_players = board_state[enemy_pieces[:, 0], enemy_pieces[:, 1], 0]
        unique_enemies = np.unique(enemy_players)

        # Create a lookup table for enemy players by id
        enemy_lookup = {
            enemy: i for i, enemy in enumerate(unique_enemies)
        }  # TODO: vectorize
        enemy_mask = np.array([enemy_lookup[enemy] for enemy in enemy_players])

        # For each unique enemy player calc the valid and other mask
        is_enemy = board_state[:, :, 0] == unique_enemies[:, None, None]
        is_piece = board_state[:, :, 1] >= 0
        own_mask = np.logical_and(is_piece, is_enemy)
        other_mask = np.logical_and(is_piece, ~is_enemy)

        moves = []
        for piece, mask_id in zip(enemy_pieces, enemy_mask):
            m = self.get_moves_position(
                board_state, piece, own_mask[mask_id, :, :], other_mask[mask_id, :, :]
            )
            if m is not None and m.shape[0] > 0:
                # Add position to move
                m = np.hstack((np.tile(piece, (m.shape[0], 1)), m))
                moves.append(m)
        if moves:
            moves = np.vstack(moves)
        return moves

    def get_masks(self, board_state, player):
        player_territory = board_state[:, :, 0] == player
        any_pieces = board_state[:, :, 1] >= 0
        own_mask = np.logical_and(any_pieces, player_territory)
        other_mask = np.logical_and(any_pieces, ~player_territory)
        return own_mask, other_mask

    def get_moves_single(self, board_state, position):
        # First calc valid and other mask
        player = board_state[position[0], position[1], 0]
        own_mask, other_mask = self.get_masks(board_state, player)
        return self.get_moves_position(board_state, position, own_mask, other_mask)

    def get_moves_position(self, board_state, position, own_mask, other_mask):
        # Get all possible moves for the piece at position
        piece_type = board_state[position[0], position[1], 1]

        if piece_type == -1:
            return None
        elif piece_type == 0:
            return self.get_pawn_moves(board_state, position, own_mask)
        elif piece_type == 1:
            return self.get_knight_moves(position, own_mask)
        elif piece_type == 2:
            return self.get_bishop_moves(board_state, position, own_mask, other_mask)
        elif piece_type == 3:
            return self.get_rook_moves(board_state, position, own_mask, other_mask)
        elif piece_type == 4:
            return self.get_queen_moves(board_state, position, own_mask, other_mask)
        elif piece_type == 5:
            return self.get_king_moves(position, own_mask)
        else:
            raise Exception("Invalid piece type", piece_type)

    def get_pawn_moves(self, board_state, position, own_mask):
        pos1, pos2 = self.all_moves[position[0]][position[1]][0]
        # This is only valid if the square is not occupied by another piece
        valid = board_state[pos1[:, 0], pos1[:, 1], 1] == -1
        pos1 = pos1.copy()[valid, :]

        # Pawn can also capture diagonaly
        valid2 = np.logical_and(
            ~own_mask[pos2[:, 0], pos2[:, 1]],
            board_state[pos2[:, 0], pos2[:, 1], 1] != -1,
        )
        pos2 = pos2.copy()[valid2, :]
        return np.vstack((pos1, pos2))

    def get_knight_moves(self, position, own_mask):
        moves = self.all_moves[position[0]][position[1]][1]
        return moves[~own_mask[moves[:, 0], moves[:, 1]], :]

    def get_bishop_moves(self, board_state, position, own_mask, other_mask):
        return self.check_diagonal_lines(board_state, position, own_mask, other_mask)

    def get_rook_moves(self, board_state, position, own_mask, other_mask):
        return self.check_straight_lines(board_state, position, own_mask, other_mask)

    def get_queen_moves(self, board_state, position, own_mask, other_mask):
        lines = self.check_straight_lines(board_state, position, own_mask, other_mask)
        lines2 = self.check_diagonal_lines(board_state, position, own_mask, other_mask)
        if lines2.size == 0:
            return lines
        elif lines.size == 0:
            return lines2
        return np.concatenate((lines, lines2))

    def get_king_moves(self, position, own_mask):
        moves = self.all_moves[position[0]][position[1]][5]
        return moves[~own_mask[moves[:, 0], moves[:, 1]], :]

    def check_straight_lines(self, board_state, position, own_mask, other_mask):
        # Check horizontal and vertical lines in all directions
        directions = [
            (1, 0, lambda x: x < self.grid_height),  # Right
            (0, -1, lambda x: x >= 0),  # Down
            (-1, 0, lambda x: x >= 0),  # Left
            (0, 1, lambda x: x < self.grid_height),  # Up
        ]

        positions = []

        for i, (x_move, y_move, validate) in enumerate(directions):
            for diff in range(1, MAX_STEPS + 1):
                x, y = position[0] + x_move * diff, position[1] + y_move * diff

                if not validate(x if x_move else y):
                    break

                target = [x, y]
                matches = np.logical_and(
                    board_state[x, y, 1] == -1, board_state[x, y, 0] != -2
                )

                if matches:
                    positions.append(target)
                elif other_mask[x, y]:
                    positions.append(target)
                    break
                elif own_mask[x, y]:
                    break
                else:
                    break

        return np.array(positions)

    def check_diagonal_lines(self, board_state, position, own_mask, other_mask):
        # Check diagonal lines in all directions
        directions = [
            (
                1,
                1,
                lambda x, y: x < self.grid_height and y < self.grid_height,
            ),  # Diagonal to down-right
            (-1, -1, lambda x, y: x >= 0 and y >= 0),  # Diagonal to up-left
            (
                -1,
                1,
                lambda x, y: x >= 0 and y < self.grid_height,
            ),  # Diagonal to up-right
            (
                1,
                -1,
                lambda x, y: x < self.grid_height and y >= 0,
            ),  # Diagonal to down-left
        ]

        positions = []

        for i, (x_move, y_move, validate) in enumerate(directions):
            for diff in range(1, MAX_STEPS + 1):
                x, y = position[0] + x_move * diff, position[1] + y_move * diff

                if not validate(x, y):
                    break

                target = [x, y]
                matches = np.logical_and(
                    board_state[x, y, 1] == -1, board_state[x, y, 0] != -2
                )

                if matches:
                    positions.append(target)
                elif other_mask[x, y]:
                    positions.append(target)
                    break
                elif own_mask[x, y]:
                    break
                else:
                    break

        return np.array(positions)

    def get_enemy_pieces(self, board_state, player, distance=-1):
        # Get all enemy pieces within distance of the bounding box of the player
        # Distance -1 means all
        if distance == -1:
            enemy_pieces = np.argwhere(
                np.logical_and(
                    board_state[:, :, 0] >= 0, board_state[:, :, 0] != player
                )
            )
            extent = np.array([[0, 0], [self.grid_height, self.grid_height]])
        else:
            bbox = np.argwhere(board_state[:, :, 0] == player)
            extent = np.array(
                [np.min(bbox, axis=0) - distance, np.max(bbox, axis=0) + distance]
            )
            extent = np.clip(extent, 0, self.grid_height)

            # Find all enemy pieces within the bounding box
            board = board_state[
                extent[0, 0] : extent[1, 0], extent[0, 1] : extent[1, 1], 0
            ]

            # Get all enemy pieces
            enemy_pieces = np.argwhere(np.logical_and(board >= 0, board != player))

            if not enemy_pieces.any():
                return None, None
            enemy_pieces += extent[0, :]
        return enemy_pieces, extent

    def precalculate_moves(self, board_state, positions):
        # Out of bounds or on water is not possible
        positions = positions.copy()
        valid = np.logical_and(
            np.logical_and(positions[:, 0] >= 0, positions[:, 0] < self.grid_height),
            np.logical_and(positions[:, 1] >= 0, positions[:, 1] < self.grid_height),
        )
        positions = positions[valid, :]
        valid = board_state[positions[:, 0], positions[:, 1], 0] != -2
        return positions[valid, :]

    def execute_move(self, board_state, move, inplace=True):
        # New territory is only the new position
        # Move is a tuple (x, y, x2, y2) starting position and end position
        if not inplace:
            board_state = board_state.copy()

        x, y, x2, y2 = move

        # Check for king capture = remove all pieces and territory
        if board_state[x2, y2, 1] == 5:
            captured_player = board_state[x2, y2, 0]
            board_state[board_state[:, :, 0] == captured_player, 1] = -1
            board_state[board_state[:, :, 0] == captured_player, 0] = -1

        board_state[x2, y2, :] = board_state[x, y, :]
        board_state[x, y, 1] = -1
        return board_state

    def execute_move_area_capture(self, board_state, move, inplace=True):
        # New territory is square between king and new position
        # Move is a tuple (x, y, x2, y2) starting position and end position
        # NOTE: This slows down search, probably due to pruning and more varied territory board score.
        if not inplace:
            board_state = board_state.copy()

        x, y, x2, y2 = move

        if board_state[x2, y2, 1] == 5:
            captured_player = board_state[x2, y2, 0]
            board_state[board_state[:, :, 0] == captured_player, 1] = -1
            board_state[board_state[:, :, 0] == captured_player, 0] = -1

        board_state[x2, y2, :] = board_state[x, y, :]
        board_state[x, y, 1] = -1

        king_loc = np.where(
            (board_state[:, :, 0] == board_state[x2, y2, 0])
            & (board_state[:, :, 1] == 5)
        )

        if len(king_loc[0]) == 0:
            return board_state

        king_x, king_y = king_loc[0][0], king_loc[1][0]

        # Get all fields in the mask that are not water (-2) or another piece
        mask = np.logical_and(board_state[:, :, 0] != -2, board_state[:, :, 1] == -1)

        # mask area from king square until new coords (x2, y2), inclusive
        if king_x < x2:
            slice_x = slice(king_x + 1, x2 + 1)
        elif king_x > x2:
            slice_x = slice(x2, king_x)
        else:
            slice_x = slice(king_x, king_x + 1)

        if king_y < y2:
            slice_y = slice(king_y + 1, y2 + 1)
        elif king_y > y2:
            slice_y = slice(y2, king_y)
        else:
            slice_y = slice(king_y, king_y + 1)

        square_mask = np.zeros_like(board_state[:, :, 0], dtype=bool)
        square_mask[slice_x, slice_y] = True

        new_territory = np.logical_and(mask, square_mask)
        board_state[new_territory, 0] = board_state[x2, y2, 0]

        return board_state

    def evaluate_board(self, board_state, player):
        pieces = board_state[:, :, 1]
        is_pieces = pieces >= 0
        pieces_score = piece_values[pieces]

        # Indices for player and enemy on the board
        is_player = board_state[:, :, 0] == player
        player_indices = np.logical_and(is_pieces, is_player)

        # Territory score
        # territory_score = np.count_nonzero(is_player)

        # Piece score
        own_pieces_score = np.sum(pieces_score[player_indices])

        # Enemy piece and territory score
        enemy_indices = np.logical_and(is_pieces, ~is_player)
        enemy_pieces_score = np.sum(pieces_score[enemy_indices])
        # enemy_territory_score = (
        #     np.count_nonzero(np.logical_and(board_state[:, :, 0] >= 0, ~is_player))
        # )

        return (
            # territory_score / 100
            own_pieces_score
            - enemy_pieces_score / 5
            # - enemy_territory_score / 500
        )

    def get_score_string(self, board_state, current_player):
        # A string with the territory for each player (with color name)
        # and the piece score for each player

        players = np.unique(board_state[:, :, 0])
        players = players[players >= 0]

        msg = ""
        for player in players:
            is_player = board_state[:, :, 0] == player
            territory_score = np.count_nonzero(is_player)
            own_pieces = np.logical_and(board_state[:, :, 1] >= 0, is_player)
            piece_score = (
                np.sum(piece_values[board_state[own_pieces, 1]]) - 100
            )  # Remove king
            is_current = "->" if player == current_player else ""
            msg += f"{is_current}{player_names[player]}: {territory_score}, {piece_score}   "
        return msg
